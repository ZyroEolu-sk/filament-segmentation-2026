"""Panoptic Quality local, fiel al notebook oficial de autoevaluación.

Fuente: reference/self-evaluation-notebook.ipynb (azimahmadzadeh). Matching por
IoU > 0.5 (sin Hungarian: con umbral 0.5 cada GT casa como mucho con una
predicción si las predicciones no se solapan entre sí).

    PQ = sum(IoU de TP) / (|TP| + 0.5|FP| + 0.5|FN|)

Las filas del CSV se agrupan por imagen con `filament_id.rsplit('_', 1)[0]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rle import decode_counts

IOU_THRESHOLD = 0.5


def image_id_of(filament_id: str) -> str:
    return filament_id.rsplit("_", 1)[0]


def overlap_matrices(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gt (n_gt, H, W) y pred (n_pred, H, W) binarias -> (iou, dice), ambas (n_gt, n_pred)."""
    g = gt.reshape(len(gt), -1).astype(np.float32)
    p = pred.reshape(len(pred), -1).astype(np.float32)
    inter = g @ p.T
    ga = g.sum(1, keepdims=True)
    pa = p.sum(1, keepdims=True).T
    union = ga + pa - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union == 0, 0.0, inter / union)
        dice = np.where(union == 0, 0.0, 2 * inter / (ga + pa))
    return iou, dice


@dataclass
class PQResult:
    pq: float
    sq: float  # media de IoU de los TP (calidad de segmentación)
    rq: float  # F1 de detección
    tp: int
    fp: int
    fn: int
    per_image: pd.DataFrame


def _masks_by_image(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for fid, counts in zip(df["filament_id"], df["segmentation_rle"]):
        out.setdefault(image_id_of(fid), []).append(decode_counts(counts))
    return {k: np.stack(v) for k, v in out.items()}


def panoptic_quality(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQResult:
    """Ambos DataFrames con columnas filament_id, segmentation_rle (formato submission)."""
    gt = _masks_by_image(gt_df)
    pred = _masks_by_image(pred_df)
    tp_ious: list[float] = []
    fp = fn = 0
    rows = []
    for img in sorted(set(gt) | set(pred)):
        g, p = gt.get(img), pred.get(img)
        n_gt = 0 if g is None else len(g)
        n_pred = 0 if p is None else len(p)
        if n_gt == 0:
            fp += n_pred
            rows.append((img, n_gt, n_pred, 0, n_pred, 0, 0.0))
            continue
        if n_pred == 0:
            fn += n_gt
            rows.append((img, n_gt, n_pred, 0, 0, n_gt, 0.0))
            continue
        iou, _ = overlap_matrices(g, p)
        hit = iou > IOU_THRESHOLD
        ious = iou[hit].tolist()
        i_fp = int((hit.sum(0) == 0).sum())
        i_fn = int((hit.sum(1) == 0).sum())
        tp_ious += ious
        fp += i_fp
        fn += i_fn
        rows.append((img, n_gt, n_pred, len(ious), i_fp, i_fn, float(np.sum(ious))))
    tp = len(tp_ious)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_ious) / den if den else 0.0
    sq = float(np.mean(tp_ious)) if tp else 0.0
    rq = tp / den if den else 0.0
    per_image = pd.DataFrame(rows, columns=["image", "n_gt", "n_pred", "tp", "fp", "fn", "sum_iou"])
    return PQResult(pq, sq, rq, tp, fp, fn, per_image)
