"""Datos: índice COCO, split por observación, caché memmap y generador de recortes.

Una observación (fichero .jpeg) puede tener hasta 3 anotaciones independientes
(image id `AAAAAA-<stem>`). Cada anotación se rasteriza a un mapa de etiquetas
uint8 (0 = fondo, k = k-ésimo filamento) y se guarda en la caché junto a la
imagen, ambos como .npy para leer recortes con memmap sin cargar 2048x2048.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = DATA / "train" / "train_images"
TEST_IMAGES = DATA / "test" / "test_images"
CACHE = ROOT / "data" / "cache"
SIZE = 2048


@dataclass
class Observation:
    stem: str  # p. ej. 20140609195854Bh
    annotations: dict[str, list[list[float]]] = field(default_factory=dict)  # image id -> polígonos

    @property
    def observatory(self) -> str:
        return self.stem[-2:]


def load_observations(path: Path = TRAIN_JSON) -> dict[str, Observation]:
    coco = json.loads(path.read_text())
    obs: dict[str, Observation] = {}
    id_to_stem = {}
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        id_to_stem[im["id"]] = stem
        obs.setdefault(stem, Observation(stem)).annotations[im["id"]] = []
    for a in coco["annotations"]:
        obs[id_to_stem[a["image_id"]]].annotations[a["image_id"]].append(a["segmentation"][0])
    return obs


def split(stems: list[str], val_frac: float = 0.15, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split por observación, estratificado por observatorio."""
    rng = np.random.default_rng(seed)
    by_obs = defaultdict(list)
    for s in sorted(stems):
        by_obs[s[-2:]].append(s)
    train, val = [], []
    for group in by_obs.values():
        group = list(rng.permutation(group))
        k = round(len(group) * val_frac)
        val += group[:k]
        train += group[k:]
    return sorted(train), sorted(val)


def rasterize(polygons: list[list[float]]) -> np.ndarray:
    """Polígonos -> mapa de etiquetas uint8. Mismo criterio que pycocotools (verificado en scripts/prepare_cache)."""
    from .rle import polygon_to_mask

    label = np.zeros((SIZE, SIZE), np.uint8)
    for k, poly in enumerate(polygons, start=1):
        label[polygon_to_mask([poly])] = k
    return label


def read_image(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert im is not None and im.shape == (SIZE, SIZE), path
    return im


def boundary_target(label: np.ndarray, width: int = 3) -> np.ndarray:
    """Zona de contacto: píxeles a menos de `width` px de dos filamentos distintos.

    No se marca el contorno entero: muchos filamentos miden 3-5 px de ancho y
    quitar su contorno los partiría en trozos. Solo se marca donde dos
    instancias se tocan o casi, que es donde hay que separarlas.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    cover = np.zeros(label.shape, np.uint8)
    for sl, i in zip(ndimage.find_objects(label), range(1, int(label.max()) + 1)):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - width, 0), sl[0].stop + width
        x0, x1 = max(sl[1].start - width, 0), sl[1].stop + width
        cover[y0:y1, x0:x1] += cv2.dilate((label[y0:y1, x0:x1] == i).astype(np.uint8), k)
    return (cover >= 2).astype(np.uint8)


def skeleton_target(fg: np.ndarray) -> np.ndarray:
    """Esqueleto de la máscara dilatado 1 px (Skeleton Recall Loss, Kirchhoff et al. ECCV'24)."""
    if not fg.any():
        return np.zeros(fg.shape, np.uint8)
    skel = skeletonize(fg).astype(np.uint8)
    return cv2.dilate(skel, np.ones((3, 3), np.uint8))


TARGET_MODES = ("random", "consensus", "majority", "consensus-ignore")


def merge_targets(labs: list[np.ndarray], mode: str) -> np.ndarray:
    """Objetivo (3, H, W) [filamento, contacto, esqueleto] a partir de los mapas de etiquetas de 1..3 anotadores.

    random    : un solo anotador (labs tiene 1 elemento): comportamiento original.
    consensus : filamento = fracción de anotadores que lo marcan (objetivo blando en [0, 1]).
    majority  : filamento = 1 donde lo marca al menos la mitad de los anotadores.
    consensus-ignore : como consensus, más un 4.º canal de validez que vale 0 en los píxeles disputados (los marca
                una minoría, 0 < fracción < 0.5) para que la loss no les enseñe un objetivo bajo (opt-in).
    En ambos casos el contacto es la unión de las zonas de contacto de cada anotador y el esqueleto se calcula
    sobre fg >= 0.5. Con dos anotadores, majority conserva los empates (es la unión semántica).
    Son hipótesis de entrenamiento: la frecuencia de anotación por sí sola no determina la decisión
    óptima para PQ, que depende también de IoU y del resto de predicciones.
    """
    if mode not in TARGET_MODES or not labs:
        raise ValueError("Modo de objetivo desconocido o lista de anotaciones vacía")
    fg = np.mean([lb > 0 for lb in labs], axis=0).astype(np.float32)
    if mode == "majority":
        fg = (fg >= 0.5).astype(np.float32)
    elif mode not in ("consensus", "consensus-ignore") and len(labs) != 1:
        raise ValueError(f"modo {mode} con {len(labs)} anotadores")
    contact = np.max([boundary_target(lb) for lb in labs], axis=0).astype(np.float32)
    skel = skeleton_target(fg >= 0.5).astype(np.float32)
    if mode == "consensus-ignore":
        valid = ~((fg > 0) & (fg < 0.5))
        return np.stack([fg, contact, skel, valid.astype(np.float32)])
    return np.stack([fg, contact, skel])


class CropGenerator(IterableDataset):
    """Generador infinito de recortes (imagen, [filamento, borde]) con aumentos al vuelo.

    - `p_focus` de los recortes se centran en un filamento al azar (hay mucho
      fondo: el filamento mediano ocupa ~1200 px de 4M).
    - Cada muestra usa un anotador al azar de la observación: el ruido entre
      anotadores actúa como regularización.
    """

    def __init__(self, stems: list[str], crop: int = 512, p_focus: float = 0.8, augment: bool = True, seed: int = 0,
                 target: str = "random", hard_negatives: dict[str, list] | None = None, p_hard: float = 0.0,
                 year_weights: dict[str, float] | None = None, augmentation: str = "standard"):
        """target: random (un anotador al azar por recorte) | consensus | majority | consensus-ignore (merge_targets).
        hard_negatives: {stem: [[y, x], ...]} (scripts/hard_negatives.py); una fracción p_hard de los recortes se
        centra en uno de esos puntos (idea #19), el resto se reparte como siempre entre filamento y azar.
        year_weights: {"2021": 2.0, ...} peso relativo de cada observación según su año (idea #7, sobremuestrear
        años escasos); los años ausentes pesan 1. None conserva el muestreo uniforme original."""
        if not 0 <= p_focus <= 1 or not 0 <= p_hard <= 1:
            raise ValueError("p_focus y p_hard deben estar entre 0 y 1")
        if target not in TARGET_MODES:
            raise ValueError(f"Objetivo desconocido: {target}")
        if not stems or not 0 < crop <= SIZE / 1.5:
            raise ValueError("Se requieren observaciones y un recorte válido")
        self.stems = stems
        self.crop = crop
        self.p_focus = p_focus
        self.augment = augment
        if augmentation not in ("standard", "solar"):
            raise ValueError("augmentation must be standard or solar")
        self.augmentation = augmentation
        self.seed = seed
        self.target = target
        allowed = set(stems)
        self.hard = {}
        for s, values in (hard_negatives or {}).items():
            if s not in allowed or not len(values):
                continue
            points = np.asarray(values, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 2:
                raise ValueError(f"Coordenadas de negativos inválidas en {s}")
            points = points[:, :2]
            if not np.isfinite(points).all() or (points < 0).any() or (points >= SIZE).any():
                raise ValueError(f"Coordenadas de negativos fuera de la imagen en {s}")
            self.hard[s] = points
        self.hard_stems = sorted(self.hard)
        self.stem_p = None
        if year_weights:
            if any(not np.isfinite(w) or w <= 0 for w in year_weights.values()):
                raise ValueError("Los pesos por año deben ser positivos")
            w = np.array([year_weights.get(s[:4], 1.0) for s in stems], dtype=np.float64)
            self.stem_p = w / w.sum()
        if p_hard > 0 and not self.hard:
            raise ValueError("p_hard > 0 requiere negativos válidos de las observaciones de entrenamiento")
        self.p_hard = p_hard if self.hard else 0.0
        index = json.loads((CACHE / "index.json").read_text())
        self.ann_ids = {s: index["annotations"][s] for s in stems}
        self.centroids = {a: np.array(c, dtype=np.float32) for s in stems for a, c in index["centroids"].items() if a in self.ann_ids[s]}

    def _crop_kind(self, u):
        """p_focus es condicional a NO elegir un negativo difícil."""
        if u < self.p_hard:
            return "hard"
        if u < self.p_hard + (1 - self.p_hard) * self.p_focus:
            return "focus"
        return "random"

    def _load(self, stem: str, ann: str):
        img = np.load(CACHE / "images" / f"{stem}.npy", mmap_mode="r")
        lab = np.load(CACHE / "labels" / f"{ann}.npy", mmap_mode="r")
        return img, lab

    def _sample(self, rng: np.random.Generator):
        u = rng.random()
        kind = self._crop_kind(u)
        if kind == "hard":  # recorte centrado en un falso positivo difícil (#19)
            stem = self.hard_stems[rng.integers(len(self.hard_stems))]
        elif self.stem_p is None:
            stem = self.stems[rng.integers(len(self.stems))]
        else:  # sobremuestreo por año (#7)
            stem = self.stems[rng.choice(len(self.stems), p=self.stem_p)]
        # Misma distribución de centros al comparar random frente a consenso: no privilegiar
        # siempre los filamentos dibujados por el primer anotador.
        anchor = self.ann_ids[stem][rng.integers(len(self.ann_ids[stem]))]
        anns = self.ann_ids[stem] if self.target != "random" else [anchor]
        img, _ = self._load(stem, anns[0])
        labs = [self._load(stem, a)[1] for a in anns]
        c = self.crop
        # contexto extra para que rotaciones/escalas no dejen bordes vacíos
        big = int(math.ceil(c * 1.5))
        cents = self.centroids[anchor]
        if kind == "hard":
            pts = self.hard[stem]
            cy, cx = pts[rng.integers(len(pts))] + rng.normal(0, c / 8, 2)
        elif kind == "focus" and len(cents):
            cy, cx = cents[rng.integers(len(cents))] + rng.normal(0, c / 6, 2)
        else:  # cualquier punto del disco solar (radio ~0.45 del lado)
            r = SIZE * 0.45 * math.sqrt(rng.random())
            t = rng.random() * 2 * math.pi
            cy, cx = SIZE / 2 + r * math.sin(t), SIZE / 2 + r * math.cos(t)
        y0 = int(np.clip(cy - big / 2, 0, SIZE - big))
        x0 = int(np.clip(cx - big / 2, 0, SIZE - big))
        im = np.array(img[y0 : y0 + big, x0 : x0 + big], dtype=np.float32) / 255.0
        lbs = [np.array(lab[y0 : y0 + big, x0 : x0 + big]) for lab in labs]

        if self.augment:
            im, lbs = self._geometric(im, lbs, rng)
            im = self._photometric(im, rng)
            if self.augmentation == "solar":
                im = self._solar_illumination(im, rng)
        off = (big - c) // 2
        im = im[off : off + c, off : off + c]
        lbs = [lb[off : off + c, off : off + c] for lb in lbs]
        # canal 2: esqueleto "tubular" (1 px de margen) para la Skeleton Recall Loss; no es una salida del modelo
        target = merge_targets(lbs, self.target)
        return torch.from_numpy(np.ascontiguousarray(im[None])), torch.from_numpy(target)

    @staticmethod
    def _solar_illumination(im, rng):
        # Smooth multiplicative field simulates uneven illumination without
        # erasing filaments or deforming their masks. Applied to train only.
        if rng.random() < 0.35:
            yy = np.linspace(-1, 1, im.shape[0], dtype=np.float32)[:, None]
            xx = np.linspace(-1, 1, im.shape[1], dtype=np.float32)[None, :]
            a, b, c = rng.uniform(-0.12, 0.12, 3)
            field = 1 + a * yy + b * xx + c * (xx**2 + yy**2 - 2/3)
            im = np.clip(im * field, 0, 1).astype(np.float32)
        return im

    @staticmethod
    def _geometric(im, lbs, rng):
        # rotación arbitraria + escala ligera: el Sol no tiene orientación preferente a escala local
        # (la misma transformación para la imagen y los mapas de todos los anotadores)
        h = im.shape[0]
        angle = rng.uniform(0, 360)
        scale = rng.uniform(0.85, 1.15)
        m = cv2.getRotationMatrix2D((h / 2, h / 2), angle, scale)
        im = cv2.warpAffine(im, m, (h, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        lbs = [cv2.warpAffine(lb, m, (h, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) for lb in lbs]
        if rng.random() < 0.5:
            im, lbs = im[:, ::-1], [lb[:, ::-1] for lb in lbs]
        return im, lbs

    @staticmethod
    def _photometric(im, rng):
        # distintos observatorios/seeing: brillo, contraste, gamma, desenfoque y ruido
        im = im * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
        im = np.clip(im, 0, 1) ** rng.uniform(0.7, 1.4)
        if rng.random() < 0.3:
            im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.5, 1.5))
        if rng.random() < 0.3:
            im = im + rng.normal(0, rng.uniform(0.005, 0.03), im.shape).astype(np.float32)
        return np.clip(im, 0, 1).astype(np.float32)

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid, torch.initial_seed() % 2**32])
        while True:
            yield self._sample(rng)
