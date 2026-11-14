"""Reparto de observaciones en folds (#14): determinista, por observación (todas las anotaciones de una imagen van
juntas) y con las observaciones simultáneas (misma marca temporal ± 1 min) en el mismo fold."""

from __future__ import annotations

from datetime import datetime

import numpy as np


def groups_by_time(stems: list[str], minutes: float = 1.0) -> list[list[str]]:
    """Agrupa stems cuyas marcas temporales distan <= minutes (observaciones simultáneas de dos observatorios)."""
    ordered = sorted(stems, key=lambda s: s[:14])
    groups, current, last = [], [], None
    for s in ordered:
        t = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
        if last is not None and (t - last).total_seconds() > minutes * 60:
            groups.append(current)
            current = []
        current.append(s)
        last = t
    if current:
        groups.append(current)
    return groups


def split_folds(stems: list[str], folds: int, seed: int) -> list[list[str]]:
    """Lista de `folds` listas de stems (los excluidos de cada fold), tamaños equilibrados, grupos simultáneos juntos."""
    groups = groups_by_time(stems)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    out = [[] for _ in range(folds)]
    sizes = [0] * folds
    for g in order:  # el grupo va al fold con menos observaciones hasta ahora
        k = int(np.argmin(sizes))
        out[k].extend(groups[g])
        sizes[k] += len(groups[g])
    return [sorted(f) for f in out]
