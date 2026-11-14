"""Recortes de GONG sin etiquetas con probabilidades docentes de ens5c (#33)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .data import CropGenerator, SIZE


class PseudoCropGenerator(IterableDataset):
    def __init__(self, data: Path, probs: Path, crop: int = 512, seed: int = 33, p_focus: float = 0.8):
        if not 0 <= p_focus <= 1 or crop > SIZE / 1.5:
            raise ValueError("p_focus/crop inválido")
        rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines()]
        self.stems = [r["stem"] for r in rows if (probs / f"{r['stem']}.npy").is_file()]
        if not self.stems:
            raise ValueError("faltan pseudo-probabilidades GONG")
        self.data, self.probs, self.crop, self.seed, self.p_focus = data, probs, crop, seed, p_focus
        self.image_dir = data / "images"
        self.image_dir.mkdir(exist_ok=True)
        self.centers = {}
        for stem in self.stems:
            image_path = self.image_dir / f"{stem}.npy"
            if not image_path.exists():
                image = cv2.imread(str(data / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (SIZE, SIZE):
                    raise ValueError(f"JPG GONG inválido: {stem}")
                np.save(image_path, image)
            p = np.load(probs / f"{stem}.npy", mmap_mode="r")
            if p.shape != (2, SIZE, SIZE):
                raise ValueError(f"probabilidades inválidas: {stem}")
            self.centers[stem] = np.argwhere(p[0, ::4, ::4] > 0.6).astype(np.int16) * 4

    def _sample(self, rng: np.random.Generator):
        stem = self.stems[rng.integers(len(self.stems))]
        centers = self.centers[stem]
        c = self.crop
        big = int(math.ceil(c * 1.5))
        if rng.random() < self.p_focus and len(centers):
            cy, cx = centers[rng.integers(len(centers))] + rng.normal(0, c / 6, 2)
        else:
            r = SIZE * 0.45 * math.sqrt(rng.random())
            t = rng.random() * 2 * math.pi
            cy, cx = SIZE / 2 + r * math.sin(t), SIZE / 2 + r * math.cos(t)
        y0 = int(np.clip(cy - big / 2, 0, SIZE - big))
        x0 = int(np.clip(cx - big / 2, 0, SIZE - big))
        img = np.load(self.image_dir / f"{stem}.npy", mmap_mode="r")
        prob = np.load(self.probs / f"{stem}.npy", mmap_mode="r")
        im = np.array(img[y0:y0 + big, x0:x0 + big], np.float32) / 255.0
        ps = np.array(prob[:, y0:y0 + big, x0:x0 + big], np.float32)
        angle = rng.uniform(0, 360)
        scale = rng.uniform(0.85, 1.15)
        m = cv2.getRotationMatrix2D((big / 2, big / 2), angle, scale)
        im = cv2.warpAffine(im, m, (big, big), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ps = np.stack([cv2.warpAffine(channel, m, (big, big), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT) for channel in ps])
        if rng.random() < 0.5:
            im, ps = im[:, ::-1], ps[:, :, ::-1]
        im = CropGenerator._photometric(im, rng)
        off = (big - c) // 2
        im = im[off:off + c, off:off + c]
        ps = ps[:, off:off + c, off:off + c]
        return torch.from_numpy(np.ascontiguousarray(im[None])), torch.from_numpy(np.ascontiguousarray(ps))

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid, torch.initial_seed() % 2**32])
        while True:
            yield self._sample(rng)
