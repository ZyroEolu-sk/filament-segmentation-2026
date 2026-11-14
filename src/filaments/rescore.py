"""Reclasificador de instancias: sustituye el filtro fijo de área/confianza del post-proceso.

El post-proceso genera candidatos sin filtrar (umbral + watershed + unión de fragmentos). Para cada uno se
calculan variables de forma, probabilidad, contraste con su entorno y posición en el disco, y un gradient
boosting estima la fracción de anotadores con los que casaría (IoU > 0.5). Se conservan los candidatos con
puntuación >= umbral, ajustado con el PQ de validación.
"""

from __future__ import annotations

from itertools import chain

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .postprocess import PostCfg, instances

OBSERVATORIES = "BCLMTU"
MIN_CANDIDATE = 50  # px; por debajo no hay filamentos anotados útiles y solo añaden ruido


def candidates(prob: np.ndarray, cfg: PostCfg) -> np.ndarray:
    """Mapa de etiquetas con los candidatos: el post-proceso de `cfg` sin su filtro final."""
    return instances(prob, PostCfg(**{**cfg.as_dict(), "min_area": MIN_CANDIDATE, "min_score": 0.0}))


def disk(im: np.ndarray) -> tuple[np.ndarray, float]:
    """Centro (x, y) y radio del disco solar: la componente más grande por encima del fondo."""
    m = (cv2.GaussianBlur(im, (0, 0), 4) > 20).astype(np.uint8)
    _, _, stats, cents = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cents[k], float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi))


def features(prob: np.ndarray, im: np.ndarray, lab: np.ndarray, stem: str) -> pd.DataFrame:
    """Una fila por candidato de `lab` (ids 1..n, en orden)."""
    (cx, cy), radius = disk(im)
    pf, pb = prob[0].astype(np.float32), prob[1].astype(np.float32)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    n = int(lab.max())
    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        pad = 10
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, lab.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, lab.shape[1])
        m = lab[y0:y1, x0:x1] == i
        ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & (lab[y0:y1, x0:x1] == 0)
        p, c, g = pf[y0:y1, x0:x1][m], pb[y0:y1, x0:x1][m], im[y0:y1, x0:x1].astype(np.float32)
        ys, xs = np.nonzero(m)
        (_, _), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(ct, True) for ct in contours)
        inside, around = g[m].mean(), g[ring].mean() if ring.any() else g[m].mean()
        r = np.hypot(xs.mean() + x0 - cx, ys.mean() + y0 - cy) / radius
        rows.append({
            "id": i, "area": m.sum(), "log_area": np.log(m.sum()),
            "p_mean": p.mean(), "p_max": p.max(), "p_p10": np.percentile(p, 10), "p_p90": np.percentile(p, 90),
            "p_frac_hi": (p > 0.8).mean(), "contact_mean": c.mean(), "contact_max": c.max(),
            "ring_p_mean": pf[y0:y1, x0:x1][ring].mean() if ring.any() else 0.0,
            "elong": max(w, h) / max(min(w, h), 1.0), "length": max(w, h), "width": m.sum() / max(max(w, h), 1.0),
            "compact": 4 * np.pi * m.sum() / max(perim, 1.0) ** 2, "n_parts": len(contours),
            "int_in": inside, "int_ring": around, "contrast": (around - inside) / max(around, 1.0),
            "int_std": g[m].std(), "r": r, "limb": float(r > 0.85),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["n_cand"] = n
    df["rank_area"] = df.area.rank(ascending=False, pct=True)
    df["rank_p"] = df.p_mean.rank(ascending=False, pct=True)
    df["disk_int"] = float(np.median(im[im > 20])) if (im > 20).any() else 0.0
    df["year"] = int(stem[:4])
    for o in OBSERVATORIES:
        df[f"obs_{o}"] = float(stem[-2] == o)
    return df


def append_label_candidates(lab: np.ndarray, cand: np.ndarray, iou_max: float = 0.5,
                            keep_frac: float = 0.5) -> tuple[np.ndarray, int]:
    """Añade instancias de un mapa de etiquetas directo, sin perder las separaciones del detector.

    Devuelve el mapa ampliado y cuántas instancias se añadieron. `cand` debe tener ids
    consecutivos 1..N, como la exportación de YOLO. No modifica `lab`.
    """
    lab = lab.astype(np.int32, copy=True)
    cand = cand.astype(np.int32, copy=False)
    n, n_lab = int(cand.max()), int(lab.max())
    if n == 0:
        return lab, 0
    inter = np.bincount((cand.astype(np.int64) * (n_lab + 1) + lab).ravel(), minlength=(n + 1) * (n_lab + 1))
    inter = inter.reshape(n + 1, n_lab + 1)[1:, 1:].astype(np.float64)
    ca = np.bincount(cand.ravel(), minlength=n + 1)[1:, None].astype(np.float64)
    la = np.bincount(lab.ravel(), minlength=n_lab + 1)[None, 1:].astype(np.float64)
    iou = inter / np.maximum(ca + la - inter, 1) if n_lab else np.zeros((n, 0))
    added = 0
    for i in range(1, n + 1):
        if n_lab and iou[i - 1].max() >= iou_max:
            continue
        m = (cand == i) & (lab == 0)
        area = int(m.sum())
        if area < MIN_CANDIDATE or area < keep_frac * ca[i - 1, 0]:
            continue
        added += 1
        lab[m] = n_lab + added
    return lab, added


def union_candidates(prob: np.ndarray, member_probs: list[np.ndarray], cfg: PostCfg,
                     iou_max: float = 0.5, keep_frac: float = 0.5,
                     direct_labs: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Candidatos del ensemble más los de cada miembro que el promedio no genera (recuperación de FN).

    Un candidato de un miembro se añade si su IoU con todos los ya presentes es < iou_max; se le quitan los píxeles
    ya asignados y se exige que conserve >= keep_frac de su área y >= MIN_CANDIDATE px. El mapa resultante es disjunto,
    así que `features`/`keep` funcionan igual. Devuelve (lab, origin) con origin[i-1] = 0 (ensemble) o k (miembro k).
    """
    lab = candidates(prob, cfg).astype(np.int32)
    origin = [0] * int(lab.max())
    sources = chain((candidates(mp, cfg) for mp in member_probs), direct_labs or ())
    for k, cand in enumerate(sources, start=1):
        lab, added = append_label_candidates(lab, cand, iou_max, keep_frac)
        origin.extend([k] * added)
    return lab, np.array(origin, np.int32)


MEMBER_FEATURES = ["mem_p_std", "mem_p_min", "mem_p_max", "mem_p_range", "mem_cov_min", "mem_cov_mean", "mem_detect_frac"]


def member_features(member_probs: list[np.ndarray], lab: np.ndarray, t_fg: float) -> pd.DataFrame:
    """Desacuerdo entre los miembros de un ensemble sobre cada candidato de `lab` (ids 1..n).

    Rasgos independientes del número y el orden de los miembros: dispersión de la probabilidad media del
    filamento dentro de la máscara, cobertura (fracción de píxeles del candidato que cada miembro pone por
    encima de t_fg) mínima y media, y fracción de miembros que «detectan» el candidato (cobertura > 0.5).
    Con un solo miembro todo es degenerado (std 0, cobertura = la del propio modelo).
    """
    n = int(lab.max())
    ids = np.arange(1, n + 1)
    means = np.zeros((len(member_probs), n), np.float32)
    covs = np.zeros((len(member_probs), n), np.float32)
    area = np.maximum(np.bincount(lab.ravel(), minlength=n + 1)[1:], 1)
    for k, prob in enumerate(member_probs):
        pf = prob[0].astype(np.float32)
        means[k] = np.bincount(lab.ravel(), weights=pf.ravel(), minlength=n + 1)[1:] / area
        covs[k] = np.bincount(lab.ravel(), weights=(pf > t_fg).ravel().astype(np.float32), minlength=n + 1)[1:] / area
    return pd.DataFrame({
        "id": ids, "mem_p_std": means.std(0), "mem_p_min": means.min(0), "mem_p_max": means.max(0),
        "mem_p_range": means.max(0) - means.min(0), "mem_cov_min": covs.min(0), "mem_cov_mean": covs.mean(0),
        "mem_detect_frac": (covs > 0.5).mean(0),
    })


def with_member_features(df: pd.DataFrame, member_probs: list[np.ndarray], lab: np.ndarray, t_fg: float) -> pd.DataFrame:
    """Añade a `df` (salida de `features`) las columnas de `member_features`, emparejando por id."""
    if df.empty:
        return df
    return df.merge(member_features(member_probs, lab, t_fg), on="id", how="left")


def with_direct_confidence(df: pd.DataFrame, direct_probs: list[np.ndarray], lab: np.ndarray) -> pd.DataFrame:
    """Confianza media máxima de los detectores sobre cada candidato, incluidos los del ensemble."""
    if df.empty:
        return df
    confidence = np.maximum.reduce([prob[0].astype(np.float32) for prob in direct_probs])
    area = np.bincount(lab.ravel(), minlength=int(lab.max()) + 1)[1:]
    mean = np.bincount(lab.ravel(), weights=confidence.ravel(), minlength=len(area) + 1)[1:] / np.maximum(area, 1)
    df["direct_conf"] = mean[df.id.to_numpy() - 1]
    return df


def keep(lab: np.ndarray, scores: np.ndarray, threshold: float, min_keep: int = 1) -> np.ndarray:
    """Conserva los candidatos con score >= threshold (al menos `min_keep`, los de mayor score) y renumera."""
    order = np.argsort(-scores)
    kept = scores >= threshold
    kept[order[:min_keep]] = True
    lut = np.zeros(lab.max() + 1, np.int32)
    ids = np.nonzero(kept)[0] + 1
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]
