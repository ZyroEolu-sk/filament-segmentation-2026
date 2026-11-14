"""Rasgos sin GT para decidir cuándo aplicar un contorno refinado."""

import cv2
import numpy as np


def gate_features(image: np.ndarray, prob: np.ndarray, seed: np.ndarray,
                  logits: np.ndarray, proposal: np.ndarray) -> dict[str, float]:
    seed = seed.astype(bool)
    proposal = proposal.astype(bool)
    image = image.astype(np.float32)
    prob = prob.astype(np.float32)
    soft = 1 / (1 + np.exp(-np.clip(logits.astype(np.float32), -20, 20)))
    area = max(int(seed.sum()), 1)
    ys, xs = np.nonzero(seed)
    height = float(np.ptp(ys) + 1) if len(ys) else 0.0
    width = float(np.ptp(xs) + 1) if len(xs) else 0.0
    ring = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool) & ~seed
    added, removed = proposal & ~seed, seed & ~proposal

    def avg(a, mask):
        return float(a[mask].mean()) if mask.any() else 0.0

    return {
        "log_area": float(np.log(area)), "bbox_long": max(height, width),
        "bbox_aspect": max(height, width) / max(min(height, width), 1),
        "seed_p_mean": avg(prob, seed), "seed_p_p10": float(np.percentile(prob[seed], 10)) if seed.any() else 0.0,
        "ring_p_mean": avg(prob, ring), "seed_int_mean": avg(image, seed),
        "ring_int_mean": avg(image, ring),
        "add_frac": float(added.sum() / area), "remove_frac": float(removed.sum() / area),
        "area_ratio": float(proposal.sum() / area),
        "added_soft_mean": avg(soft, added), "removed_soft_mean": avg(soft, removed),
        "seed_soft_mean": avg(soft, seed), "ring_soft_mean": avg(soft, ring),
    }
