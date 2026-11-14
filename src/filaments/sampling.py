"""Selección reproducible de observaciones por año y observatorio."""

from collections import defaultdict

import numpy as np


def select_observations(stems, limit=0, seed=0, strategy="stratified"):
    """Muestra proporcional con cuotas por resto mayor; 0 selecciona todo."""
    stems = sorted(stems)
    if len(set(stems)) != len(stems) or limit < 0:
        raise ValueError("Observaciones duplicadas o límite negativo")
    if strategy not in ("stratified", "chronological"):
        raise ValueError(f"Estrategia desconocida: {strategy}")
    if not limit or limit >= len(stems):
        return stems
    if strategy == "chronological":
        return stems[:limit]
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for stem in stems:
        groups[(stem[:4], stem[-2:])].append(stem)
    keys = sorted(groups)
    quotas = np.array([len(groups[k]) * limit / len(stems) for k in keys])
    counts = np.floor(quotas).astype(int)
    # Desempate aleatorio reproducible, sin preferir los años antiguos.
    order = np.lexsort((rng.random(len(keys)), -(quotas - counts)))
    counts[order[:limit - counts.sum()]] += 1
    selected = []
    for key, count in zip(keys, counts):
        selected.extend(rng.choice(groups[key], count, replace=False).tolist())
    return sorted(selected)
