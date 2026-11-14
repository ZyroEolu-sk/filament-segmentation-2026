"""Refinador local de contornos para candidatos pequeños, sin cambiar sus identificadores."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PATCH = 256
MAX_BOX = 192


def patch_origin(sl: tuple[slice, slice], shape: tuple[int, int], size: int = PATCH) -> tuple[int, int] | None:
    """Origen de un recorte nativo que contiene toda la caja con ≥32 px de margen nominal."""
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if max(h, w) > size - 64:
        return None
    cy, cx = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
    return (int(np.clip(cy - size // 2, 0, shape[0] - size)),
            int(np.clip(cx - size // 2, 0, shape[1] - size)))


def patch_input(image: np.ndarray, clahe: np.ndarray, prob: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Imagen, contraste local, probabilidad y máscara semilla, todos H×W."""
    return np.ascontiguousarray(np.stack([image / 255.0, clahe / 255.0,
                                           prob.astype(np.float32), seed.astype(np.float32)]).astype(np.float32))


def local_contrast(image: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Refiner(nn.Module):
    """U-Net pequeño que aprende una corrección residual a la máscara semilla.

    La última capa arranca a cero: antes de entrenar, el modelo devuelve la semilla.
    """

    def __init__(self, width: int = 24):
        super().__init__()
        self.e1 = Block(4, width)
        self.e2 = Block(width, width * 2)
        self.e3 = Block(width * 2, width * 4)
        self.mid = Block(width * 4, width * 8)
        self.d3 = Block(width * 12, width * 4)
        self.d2 = Block(width * 6, width * 2)
        self.d1 = Block(width * 3, width)
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        z = self.mid(F.avg_pool2d(c, 2))
        z = self.d3(torch.cat([F.interpolate(z, size=c.shape[-2:], mode="bilinear", align_corners=False), c], 1))
        z = self.d2(torch.cat([F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        z = self.d1(torch.cat([F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        seed_logit = x[:, 3:4] * 4.0 - 2.0
        return seed_logit + self.out(z)


def refine_mask(seed: np.ndarray, logits: np.ndarray, max_change_px: int = 8) -> np.ndarray:
    """Aplica cambios solo cerca del candidato y rechaza cambios extremos de área."""
    seed = seed.astype(bool)
    grown = cv2.dilate(seed.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max_change_px + 1,) * 2)).astype(bool)
    proposal = (logits > 0) & grown
    area = int(seed.sum())
    if area == 0 or proposal.sum() < max(50, 0.5 * area) or proposal.sum() > 1.5 * area:
        return seed
    return proposal
