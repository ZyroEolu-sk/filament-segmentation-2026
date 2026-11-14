"""Selección de dispositivo (CUDA en el portátil, MPS en el Mac M5, CPU como último recurso) y precisión mixta.

CUDA: autocast fp16 + GradScaler, como siempre. MPS: autocast bf16 sin scaler (bf16 no necesita escalado y fp16 en
MPS ha dado problemas de desbordamiento). CPU: sin autocast. `pick(None)` elige el primero disponible.
"""

from __future__ import annotations

import contextlib

import torch


def pick(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast(device: str):
    kind = device.split(":")[0]
    if kind == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if kind == "mps":
        return torch.autocast("mps", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def grad_scaler(device: str) -> torch.amp.GradScaler:
    """Scaler activo solo en CUDA (fp16); en MPS/CPU es un no-op con la misma interfaz."""
    return torch.amp.GradScaler("cuda", enabled=device.split(":")[0] == "cuda")


def empty_cache(device: str) -> None:
    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def synchronize(device: str) -> None:
    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
