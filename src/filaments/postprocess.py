"""Probabilidades -> instancias de filamento, y PQ rápido sobre mapas de etiquetas.

Pasos (cada umbral se ajusta con el PQ de validación, scripts/tune_postprocess.py):
1. máscara = p_fil > t_fg
2. semillas = máscara & p_contacto < t_b (zona donde dos filamentos se tocan);
   cada componente es un marcador y el watershed sobre -p_fil reparte la zona
   de contacto entre los filamentos vecinos.
3. fragmentos de *distintas* componentes a menos de `merge_dist` px se unen:
   el GT es siempre un único polígono, y un filamento partido en dos trozos
   cuenta como FP + FN en vez de TP.
4. se descartan instancias pequeñas o de baja confianza (un FP cuesta 0.5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


@dataclass(frozen=True)
class PostCfg:
    t_fg: float = 0.5
    t_b: float = 0.5
    split: bool = True
    merge_dist: int = 0
    min_area: int = 150
    min_score: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def instances(prob: np.ndarray, cfg: PostCfg) -> np.ndarray:
    """prob (2, H, W) -> mapa de etiquetas int32 (0 = fondo)."""
    pf = prob[0].astype(np.float32)
    mask = pf > cfg.t_fg
    comps, n_comp = ndi.label(mask)
    if n_comp == 0:
        return comps
    if cfg.split:
        seeds, _ = ndi.label(mask & (prob[1] < cfg.t_b))
        lab = watershed(-pf, seeds, mask=mask)
        # componentes sin semilla (todo borde): se quedan como instancia propia
        orphan = mask & (lab == 0)
        if orphan.any():
            extra, _ = ndi.label(orphan)
            lab[orphan] = extra[orphan] + lab.max()
    else:
        lab = comps
    if cfg.merge_dist > 0:
        lab = _merge_fragments(lab, comps, cfg.merge_dist)
    return _filter(lab, pf, cfg)


def _merge_fragments(lab: np.ndarray, comps: np.ndarray, dist: int) -> np.ndarray:
    """Une segmentos de componentes distintas separadas por menos de `dist` px (union-find)."""
    ids = np.unique(lab)[1:]
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dist + 1, 2 * dist + 1))
    objs = ndi.find_objects(lab)
    for i in ids:
        sl = objs[i - 1]
        y0, y1 = max(sl[0].start - dist, 0), min(sl[0].stop + dist, lab.shape[0])
        x0, x1 = max(sl[1].start - dist, 0), min(sl[1].stop + dist, lab.shape[1])
        win = lab[y0:y1, x0:x1]
        own = win == i
        grown = cv2.dilate(own.astype(np.uint8), k).astype(bool)
        my_comp = comps[y0:y1, x0:x1][own][0]
        near = grown & (win > 0) & (win != i) & (comps[y0:y1, x0:x1] != my_comp)
        for j in np.unique(win[near]):
            parent[find(j)] = find(i)
    lut = np.zeros(lab.max() + 1, np.int32)
    for i in ids:
        lut[i] = find(i)
    return lut[lab]


def _filter(lab: np.ndarray, pf: np.ndarray, cfg: PostCfg) -> np.ndarray:
    ids, area = np.unique(lab, return_counts=True)
    score = ndi.mean(pf, lab, ids)
    keep = (ids > 0) & (area >= cfg.min_area) & (score >= cfg.min_score)
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[ids[keep]] = np.arange(1, keep.sum() + 1)
    return lut[lab]


def pq_counts(gt: np.ndarray, pred: np.ndarray) -> tuple[float, int, int, int]:
    """(suma IoU de TP, TP, FP, FN) entre dos mapas de etiquetas; mismo criterio IoU > 0.5 que el oficial."""
    n_gt, n_pred = int(gt.max()), int(pred.max())
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0, n_pred, n_gt
    inter = np.bincount((gt.astype(np.int64) * (n_pred + 1) + pred).ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    inter = inter.reshape(n_gt + 1, n_pred + 1)[1:, 1:].astype(np.float64)
    ga = np.bincount(gt.ravel(), minlength=n_gt + 1)[1:, None]
    pa = np.bincount(pred.ravel(), minlength=n_pred + 1)[None, 1:]
    union = ga + pa - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    hit = iou > 0.5
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())


def pq(counts: list[tuple[float, int, int, int]]) -> float:
    s, tp, fp, fn = (sum(c[i] for c in counts) for i in range(4))
    den = tp + 0.5 * fp + 0.5 * fn
    return s / den if den else 0.0
