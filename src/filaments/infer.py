"""Inferencia a resolución completa (2048x2048) con TTA de volteos."""

from __future__ import annotations

import numpy as np
import torch

from .device import autocast, pick

FLIPS = [(), (-1,), (-2,), (-2, -1)]


@torch.no_grad()
@torch.inference_mode()
def predict(model: torch.nn.Module, image: np.ndarray, tta: bool = True, device: str | None = None) -> np.ndarray:
    """image uint8 (H, W) -> probabilidades float16 (2, H, W): filamento, borde.

    inference_mode: sin él, cada pasada a 2048² guardaba el grafo de autograd (varios GB de pico); en WSL/WDDM ese
    pico no da OOM sino que pagina VRAM a RAM y el entrenamiento que evalúa queda ~5x más lento (2026-09-26)."""
    device = pick(device)
    tile = getattr(model, "inference_tile", 0)
    if tile:
        return predict_tiled(model, image, tile, getattr(model, "inference_overlap", 128), tta, device)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None, None].to(device).contiguous(memory_format=torch.channels_last)
    acc = torch.zeros((2, *image.shape), device=device)
    flips = FLIPS if tta else FLIPS[:1]
    with autocast(device):
        for dims in flips:
            xi = torch.flip(x, dims) if dims else x
            p = torch.sigmoid(model(xi).float())
            acc += (torch.flip(p, dims) if dims else p)[0]
    return (acc / len(flips)).half().cpu().numpy()


@torch.inference_mode()
def predict_tiled(model, image, tile=512, overlap=128, tta=True, device="cpu"):
    """Blend overlapping native-resolution tiles; invert each TTA before blending.

    Accumulate on CPU to bound GPU memory independently of full image size.
    Positive Hann weights avoid seams and uncovered image borders.
    """
    if tile <= 0 or tile % 32 or not 0 <= overlap < tile:
        raise ValueError("tile must be a positive multiple of 32, 0 <= overlap < tile")
    height, width = image.shape
    padded = np.pad(image, ((0, max(0, tile-height)), (0, max(0, tile-width))), mode="edge")
    h, w = padded.shape
    def starts(n):
        return sorted(set(range(0, n-tile+1, tile-overlap)) | {n-tile})
    axis = np.maximum(np.hanning(tile).astype(np.float32), 0.05)
    weight = axis[:, None] * axis[None, :]
    acc = np.zeros((2, h, w), np.float32)
    norm = np.zeros((h, w), np.float32)
    flips = FLIPS if tta else FLIPS[:1]
    for y in starts(h):
        for x in starts(w):
            patch = np.ascontiguousarray(padded[y:y+tile, x:x+tile], dtype=np.float32) / 255
            inp = torch.from_numpy(patch)[None, None].to(device)
            total = torch.zeros((1, 2, tile, tile), device=device)
            with autocast(pick(device)):
                for dims in flips:
                    prob = model(torch.flip(inp, dims) if dims else inp).float().sigmoid()
                    total += torch.flip(prob, dims) if dims else prob
            acc[:, y:y+tile, x:x+tile] += total[0].cpu().numpy() / len(flips) * weight
            norm[y:y+tile, x:x+tile] += weight
    return (acc[:, :height, :width] / norm[:height, :width]).astype(np.float16)
