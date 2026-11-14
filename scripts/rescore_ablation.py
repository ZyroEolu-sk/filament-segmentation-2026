"""Ablación CPU sobre candidatos guardados con ajuste de umbral anidado.

No es CV independiente del segmentador: checkpoints y post-proceso ya usaron val.
Separa únicamente el ajuste del reclasificador/umbral de su evaluación exterior.
No modifica model.joblib ni genera envíos. Uso: --name r50-long.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from rescore_train import ROOT, kept_by_threshold, make_model, pq_from
from filaments.validation import compare_reports, public_sample_diagnostic


def observation_counts(df, gts, kept):
    """Sumas por observación; incluye anotadores e imágenes sin candidatos."""
    result = {stem: np.array([0., 0., 0., sum(ns)]) for stem, ns in gts.items()}
    for stem, ids in df.groupby("stem").indices.items():
        m = np.stack(df.match.values[ids])[kept[ids]]
        hit = m > 0
        tp = int(hit.sum())
        result[stem] = np.array([m.sum(), tp, m.size - tp, sum(gts[stem]) - tp])
    return result


def score(counts):
    c = np.asarray(counts).sum(axis=0)
    den = c[1] + .5 * (c[2] + c[3])
    return float(c[0] / den) if den else 0.


def threshold(df, gts, scores, min_keep):
    # Misma rejilla y presupuesto para ambos objetivos; incluye conservar/descartar todo.
    best = (-1., None)
    for t in np.linspace(0, 1, 81):
        kept = kept_by_threshold(df, scores, t, min_keep)
        counts = observation_counts(df, gts, kept)
        value = score(list(counts.values()))
        if value > best[0]:
            best = value, float(t)
    return best[1]


def subset(df, gts, stems):
    return df[df.stem.isin(stems)].reset_index(drop=True), {s: gts[s] for s in stems}


def outcome(frame, target):
    return frame.y.to_numpy() if target == "match" else np.array([np.mean(m) for m in frame.match])


def nested(df, gts, train, features, target, folds, seed):
    stems = np.array(sorted(gts))
    pred = np.zeros(len(df))
    kept = {k: np.zeros(len(df), dtype=bool) for k in (0, 1)}
    records = []
    for fold, (dev_i, test_i) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(stems)):
        dev_stems, test_stems = stems[dev_i], stems[test_i]
        dev, dev_gts = subset(df, gts, dev_stems)
        calibration = np.zeros(len(dev))
        for inner, (fit_i, cal_i) in enumerate(KFold(3, shuffle=True, random_state=seed + fold).split(dev_stems)):
            fit_stems, cal_stems = dev_stems[fit_i], dev_stems[cal_i]
            assert not set(fit_stems) & set(cal_stems)
            assert not set(test_stems) & set(fit_stems)
            fit = pd.concat([train, dev[dev.stem.isin(fit_stems)]], ignore_index=True)
            mask = dev.stem.isin(cal_stems).to_numpy()
            model = make_model(seed + inner)
            model.fit(fit[features].to_numpy(), outcome(fit, target))
            calibration[mask] = model.predict(dev.loc[mask, features].to_numpy())
        fit = pd.concat([train, dev], ignore_index=True)
        model = make_model(seed)
        model.fit(fit[features].to_numpy(), outcome(fit, target))
        ids = np.flatnonzero(df.stem.isin(test_stems))
        pred[ids] = model.predict(df.iloc[ids][features].to_numpy())
        part, _ = subset(df, gts, test_stems)
        thresholds = {}
        for k in (0, 1):
            t = threshold(dev, dev_gts, calibration, k)
            kept[k][ids] = kept_by_threshold(part, pred[ids], t, k)
            thresholds[str(k)] = t
        records.append({"fold": fold, "development_stems": dev_stems.tolist(),
                        "evaluation_stems": test_stems.tolist(), "thresholds": thresholds})
        print(f"{target} {'year' if 'year' in features else 'no_year'} fold {fold + 1}/{folds}: {thresholds}", flush=True)
    return pred, kept, records


def bootstrap_delta(a, b, seed):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(a), (2000, len(a)))
    def scores(values):
        c = values[ids].sum(axis=1)
        den = c[:, 1] + .5 * (c[:, 2] + c[:, 3])
        return np.divide(c[:, 0], den, out=np.zeros(len(c)), where=den > 0)
    return np.quantile(scores(a) - scores(b), [.025, .975]).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--train-dataset", type=Path, help="candidatos alternativos para comparar muestreos")
    ap.add_argument("--rescore-dir", type=Path, help="directorio con dataset.pkl/train_dataset.pkl (por defecto runs/<name>/rescore)")
    ap.add_argument("--diagnose-report", type=Path, help="analizar un informe existente, sin entrenar")
    ap.add_argument("--compare-report", type=Path, help="informe de referencia para delta pareado")
    ap.add_argument("--variant", default="match_year")
    ap.add_argument("--baseline-variant")
    ap.add_argument("--diagnostic-min-keep", type=int, choices=[0, 1], default=1)
    ap.add_argument("--public-size", type=int, default=90)
    ap.add_argument("--bootstrap-repetitions", type=int, default=20000)
    ap.add_argument("--reference-pq", type=float)
    args = ap.parse_args()
    if args.diagnose_report:
        if not args.output_dir:
            ap.error("--diagnose-report requiere --output-dir")
        report = json.loads(args.diagnose_report.read_text())
        result = {"sampling": public_sample_diagnostic(report, args.variant, args.diagnostic_min_keep,
                  args.public_size, args.bootstrap_repetitions, args.seed, args.reference_pq)}
        inputs = [args.diagnose_report]
        if args.compare_report:
            baseline = json.loads(args.compare_report.read_text())
            result["comparison"] = compare_reports(report, baseline, args.variant, args.baseline_variant,
                                      args.diagnostic_min_keep, args.bootstrap_repetitions, args.seed)
            inputs.append(args.compare_report)
        result["input_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if not args.name or args.compare_report:
        ap.error("El entrenamiento requiere --name; --compare-report requiere --diagnose-report")
    if args.train_dataset and not args.output_dir:
        ap.error("--train-dataset requiere --output-dir para conservar la comparación de referencia")
    run = ROOT / "runs" / args.name
    out = args.output_dir or run / "rescore_ablation"
    out.mkdir(parents=True, exist_ok=True)
    rescore_dir = args.rescore_dir or run / "rescore"
    df, gts = pd.read_pickle(rescore_dir / "dataset.pkl")
    train_path = args.train_dataset or rescore_dir / "train_dataset.pkl"
    train, train_gts = pd.read_pickle(train_path)
    assert not set(gts) & set(train_gts), "Train y validación comparten observaciones"
    feats = [c for c in df if c not in ("stem", "id", "match", "y")]
    report = {"name": args.name, "seed": args.seed, "folds": args.folds,
              "limitation": "CV anidada del reclasificador, no del segmentador/post-proceso; train in-sample.",
              "train_stems": sorted(train_gts), "validation_stems": sorted(gts),
              "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (rescore_dir / "dataset.pkl", train_path)},
              "variants": {}}
    reference = None
    for target, use_year in (("match", True), ("expected_iou", True), ("match", False), ("expected_iou", False)):
        label = f"{target}_{'year' if use_year else 'no_year'}"
        features = [c for c in feats if use_year or c != "year"]
        with threadpool_limits(limits=args.threads):
            pred, kept, fold_records = nested(df, gts, train, features, target, args.folds, args.seed)
        variant = {"folds": fold_records, "features": features, "policies": {}}
        for k in (0, 1):
            counts = observation_counts(df, gts, kept[k])
            array = np.stack([counts[s] for s in sorted(gts)])
            pq = score(array)
            if reference is None and k == 1:
                reference = array
            result = {"pq": pq, "counts": array.sum(axis=0).tolist(),
                      "per_observation": {s: c.tolist() for s, c in counts.items()}}
            if k == 1:
                result["delta_vs_match_year"] = pq - score(reference)
                result["paired_bootstrap_95"] = bootstrap_delta(array, reference, args.seed)
            variant["policies"][str(k)] = result
            print(f"{label} min_keep={k}: PQ {pq:.6f}", flush=True)
            # Comprobación independiente contra el evaluador previo de candidatos.
            assert np.isclose(pq, pq_from(df, gts, kept[k])[0])
        report["variants"][label] = variant
        pd.DataFrame({"stem": df.stem, "id": df.id, "score": pred,
                      "keep_0": kept[0], "keep_1": kept[1]}).to_csv(out / f"{label}.csv", index=False)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Informe: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
