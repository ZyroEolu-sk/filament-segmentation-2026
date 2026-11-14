"""Diagnósticos de PQ por observación, manteniendo juntos todos sus anotadores."""

import numpy as np


def pq_totals(totals):
    totals = np.asarray(totals, dtype=float)
    den = totals[..., 1] + .5 * (totals[..., 2] + totals[..., 3])
    return np.divide(totals[..., 0], den, out=np.zeros_like(den), where=den > 0)


def report_counts(report, variant="match_year", min_keep=1):
    part = report["variants"][variant]
    counts = part["policies"][str(min_keep)]["per_observation"]
    stems = sorted(report["validation_stems"])
    if len(set(stems)) != len(stems) or set(stems) != set(counts):
        raise ValueError("Las observaciones del informe y sus conteos no coinciden")
    array = np.asarray([counts[s] for s in stems], dtype=float)
    if array.shape != (len(stems), 4) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Conteos PQ inválidos")
    folds = part["folds"]
    seen = []
    for fold in folds:
        dev, test = fold["development_stems"], fold["evaluation_stems"]
        if set(dev) & set(test) or set(dev) | set(test) != set(stems):
            raise ValueError("Folds solapados o incompletos")
        seen.extend(test)
    if sorted(seen) != stems or len(folds) != report["folds"]:
        raise ValueError("Cada observación debe evaluarse una vez")
    if set(report["train_stems"]) & set(stems):
        raise ValueError("Train y validación se solapan")
    return stems, array, folds


def resample_scores(a, size, repetitions=20000, seed=20260926, b=None):
    """Remuestreo con reemplazo, por lotes pequeños; delta pareado si se pasa b."""
    if size < 1 or repetitions < 2 or len(a) == 0:
        raise ValueError("Tamaño de muestra/repeticiones no válido")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for start in range(0, repetitions, 1000):
        stop = min(start + 1000, repetitions)
        ids = rng.integers(0, len(a), (stop - start, size))
        v = pq_totals(a[ids].sum(axis=1))
        if b is not None:
            v -= pq_totals(b[ids].sum(axis=1))
        values[start:stop] = v
    return values


def public_sample_diagnostic(report, variant="match_year", min_keep=1, size=90,
                             repetitions=20000, seed=20260926, reference_pq=None):
    stems, a, _ = report_counts(report, variant, min_keep)
    v = resample_scores(a, size, repetitions, seed)
    result = {"variant": variant, "min_keep": min_keep, "validation_observations": len(stems),
              "sample_observations": size, "repetitions": repetitions, "seed": seed,
              "pq_full_validation": float(pq_totals(a.sum(axis=0))),
              "bootstrap_mean": float(v.mean()), "bootstrap_std": float(v.std(ddof=1)),
              "bootstrap_95": np.quantile(v, [.025, .975]).tolist(),
              "limitation": "Distribución empírica de val con reemplazo; no separa sesgo de varianza ni valida el pipeline completo."}
    if reference_pq is not None:
        result.update(reference_pq=reference_pq,
                      empirical_fraction_at_or_below_reference=float((v <= reference_pq).mean()),
                      reference_note="Referencia externa descriptiva; si está redondeada no es el PQ exacto. No es un p-valor ni una probabilidad de ganar.")
    return result


def compare_reports(candidate, baseline, variant="match_year", baseline_variant=None,
                    min_keep=1, repetitions=20000, seed=20260926):
    baseline_variant = baseline_variant or variant
    stems, a, folds_a = report_counts(candidate, variant, min_keep)
    stems_b, b, folds_b = report_counts(baseline, baseline_variant, min_keep)
    if stems != stems_b or candidate["seed"] != baseline["seed"]:
        raise ValueError("Comparación no pareada: observaciones o semillas diferentes")
    split_signature = lambda folds: [(f["fold"], sorted(f["development_stems"]), sorted(f["evaluation_stems"]))
                                      for f in folds]
    if split_signature(folds_a) != split_signature(folds_b):
        raise ValueError("Los informes no usan los mismos folds exteriores")
    if not np.array_equal(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]):
        raise ValueError("Número de GT por observación distinto entre informes")
    v = resample_scores(a, len(stems), repetitions, seed, b=b)
    lower, upper = np.quantile(v, [.025, .975])
    return {"candidate": candidate["name"], "baseline": baseline["name"],
            "variant": variant, "baseline_variant": baseline_variant, "min_keep": min_keep,
            "candidate_pq": float(pq_totals(a.sum(axis=0))),
            "baseline_pq": float(pq_totals(b.sum(axis=0))),
            "delta": float(pq_totals(a.sum(axis=0)) - pq_totals(b.sum(axis=0))),
            "paired_bootstrap_95": [float(lower), float(upper)], "repetitions": repetitions,
            "seed": seed, "positive_lower_bound": bool(lower > 0),
            "limitation": "Intervalo condicional a predicciones/folds; no corrige selección previa ni comparaciones múltiples. No autoriza un envío automáticamente."}
