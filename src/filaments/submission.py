"""Escritura y validación del CSV de envío.

Validaciones basadas en lo que el evaluador comprueba o penaliza: todas las
imágenes de test presentes, ids únicos, RLE decodificable a 2048x2048, sin
máscaras vacías. Las predicciones de una misma imagen no deben solaparse (con
solapes una GT puede casar con dos predicciones y contar FP/TP raros).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rle import SIZE, decode_counts, encode_counts


def build_submission(preds: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """preds: image_id (sin .jpeg) -> lista de máscaras binarias 2048x2048."""
    rows = []
    for img, masks in preds.items():
        for k, m in enumerate((m for m in masks if m.any()), start=1):
            rows.append((f"{img}_{k}", encode_counts(m)))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def validate(df: pd.DataFrame, test_dir: Path) -> list[str]:
    errors = []
    expected = {p.stem for p in Path(test_dir).glob("*.jpeg")}
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        errors.append(f"columnas inesperadas: {list(df.columns)}")
    if df["filament_id"].duplicated().any():
        errors.append("filament_id duplicados")
    present = {f.rsplit("_", 1)[0] for f in df["filament_id"]}
    if missing := expected - present:
        errors.append(f"{len(missing)} imágenes de test sin predicción")
    if extra := present - expected:
        errors.append(f"{len(extra)} ids que no son de test: {sorted(extra)[:3]}")
    for fid, counts in zip(df["filament_id"], df["segmentation_rle"]):
        try:
            m = decode_counts(counts)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fid}: RLE inválido ({e})")
            continue
        if m.shape != (SIZE, SIZE):
            errors.append(f"{fid}: forma {m.shape}")
        elif not m.any():
            errors.append(f"{fid}: máscara vacía")
    return errors


def write(df: pd.DataFrame, path: Path, test_dir: Path) -> None:
    if errors := validate(df, test_dir):
        raise ValueError("submission inválida:\n" + "\n".join(errors))
    df.to_csv(path, index=False)
