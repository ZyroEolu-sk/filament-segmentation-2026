"""RLE COCO comprimido <-> máscara binaria. Tamaño fijo 2048x2048 (el CSV solo lleva `counts`)."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

SIZE = 2048


def encode_counts(mask: np.ndarray) -> str:
    """Máscara (H, W) bool/uint8 -> string `counts` para la columna segmentation_rle."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def decode_counts(counts: str, size: int = SIZE) -> np.ndarray:
    return mask_util.decode({"size": [size, size], "counts": counts.encode("ascii")}).astype(bool)


def polygon_to_mask(segmentation: list[list[float]], h: int = SIZE, w: int = SIZE) -> np.ndarray:
    rles = mask_util.frPyObjects(segmentation, h, w)
    return mask_util.decode(mask_util.merge(rles)).astype(bool)
