"""Entrena y valida el reclasificador de instancias (filaments.rescore) sobre las imágenes de validación.

El modelo de segmentación no ha visto val, así que sus candidatos ahí son representativos del test. Con
solo 106 imágenes, el reclasificador se evalúa con validación cruzada por imagen (GroupKFold repetido):
cada imagen se puntúa con un modelo que no la ha visto, y el umbral se elige con esas puntuaciones.
Por ello el PQ seleccionado es exploratorio, no una evaluación independiente del umbral.
Para comparar objetivos con selección anidada, usar scripts/rescore_ablation.py.

Con --fit train se añaden los candidatos de las imágenes de train (más datos, pero el modelo de
segmentación ya las vio y está más seguro en ellas) y se comparan, siempre evaluando en val:
  A  reclasificador entrenado solo con train
  B  train + los folds de val que no se evalúan (validación cruzada por imagen en val)
y se guarda el mejor de {solo val, A, B}.

Uso: ~/venvs/ml/bin/python scripts/rescore_train.py --name r50-base [--fit train]
Salida: runs/<name>/rescore/{dataset.pkl, train_dataset.pkl, model.joblib}; runs/<name>/train_probs/

Los límites de train se muestrean por año/observatorio; --train-sampling chronological
reproduce los experimentos antiguos. --output-dir permite conservar el modelo vigente.
Por defecto se calibra min_keep=1, como en predict_test.py; min_keep=0 es diagnóstico.
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filaments.device import empty_cache, pick  # noqa: E402
from filaments.data import CACHE  # noqa: E402
from filaments.postprocess import PostCfg  # noqa: E402
from filaments.rescore import candidates, features, union_candidates, with_direct_confidence, with_member_features  # noqa: E402
from filaments.sampling import select_observations  # noqa: E402


def one_image(args):
    """Variables de cada candidato + IoU con su pareja (> 0.5) para cada anotador; 0 si no casa."""
    stem, anns, prob_path, cfg, member_paths, mode, direct_paths, direct_confidence = (*args, None, "features", None, False)[:8]
    prob = np.load(prob_path).astype(np.float32)
    im = np.load(CACHE / "images" / f"{stem}.npy")
    member_probs = [np.load(p).astype(np.float32) for p in member_paths] if member_paths else []
    if mode == "union":
        direct_labs = [np.load(p) for p in direct_paths] if direct_paths else None
        lab, origin = union_candidates(prob, member_probs, cfg, direct_labs=direct_labs)
        df = features(prob, im, lab, stem)
        if not df.empty:
            df["from_member"] = (origin[df.id.values - 1] > 0).astype(float)
    else:
        lab = candidates(prob, cfg)
        df = features(prob, im, lab, stem)
        if member_probs:
            df = with_member_features(df, member_probs, lab, cfg.t_fg)
    if direct_confidence and direct_paths:
        direct_probs = [np.load(Path(str(p).replace(".lab.npy", ".npy")), mmap_mode="r") for p in direct_paths]
        df = with_direct_confidence(df, direct_probs, lab)
    n_gt = []
    matches = np.zeros((int(lab.max()), len(anns)))
    for k, a in enumerate(anns):
        gt = np.load(CACHE / "labels" / f"{a}.npy")
        n_gt.append(int(gt.max()))
        if gt.max() and lab.max():
            n_p = int(lab.max())
            inter = np.bincount((gt.astype(np.int64) * (n_p + 1) + lab).ravel(), minlength=(n_gt[-1] + 1) * (n_p + 1))
            inter = inter.reshape(n_gt[-1] + 1, n_p + 1)[1:, 1:].astype(np.float64)
            ga = np.bincount(gt.ravel(), minlength=n_gt[-1] + 1)[1:, None]
            pa = np.bincount(lab.ravel(), minlength=n_p + 1)[None, 1:]
            iou = inter / np.maximum(ga + pa - inter, 1)
            best = iou.max(0)
            matches[:, k] = np.where(best > 0.5, best, 0.0)  # IoU > 0.5 es emparejamiento único
    if not df.empty:
        df.insert(0, "stem", stem)
        df["match"] = list(matches[df.id.values - 1])
        df["y"] = (matches[df.id.values - 1] > 0).mean(1)
    return df, {"stem": stem, "n_gt": n_gt}


def pq_from(df: pd.DataFrame, gts: dict, kept: np.ndarray) -> tuple[float, int, int, int]:
    """PQ con un subconjunto de candidatos. Los emparejamientos IoU > 0.5 no dependen del resto de instancias,
    así que basta con sumar los de los candidatos conservados."""
    s = tp = fp = fn = 0
    for stem, idx in df.groupby("stem").indices.items():
        m = np.stack(df.match.values[idx])[kept[idx]]  # (conservados, anotadores)
        for k, n in enumerate(gts[stem]):
            hit = m[:, k] > 0 if len(m) else np.zeros(0, bool)
            s += m[hit, k].sum()
            tp += hit.sum()
            fp += (~hit).sum()
            fn += n - hit.sum()
    for stem, n_list in gts.items():  # imágenes sin ningún candidato
        if stem not in df.stem.values:
            fn += sum(n_list)
    return s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9), tp, fp, fn


def kept_by_threshold(df, scores, t, min_keep):
    kept = scores >= t
    if min_keep:
        for _, idx in df.groupby("stem").indices.items():
            top = idx[np.argsort(-scores[idx])[:min_keep]]
            kept[top] = True
    return kept


def make_model(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=seed)


def ensure_probs(run: Path, stems: list[str], prob_dir: Path, device: str = "auto"):
    """Predice (con TTA) las imágenes que falten en prob_dir; se puede interrumpir y retomar."""
    todo = [s for s in stems if not (prob_dir / f"{s}.npy").exists()]
    if not todo:
        return
    import torch

    from filaments.infer import predict
    from filaments.model import build

    prob_dir.mkdir(exist_ok=True)
    device = pick(device)
    model = build(json.loads((run / "config.json").read_text()), pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    for i, s in enumerate(todo, 1):
        tmp = prob_dir / f"{s}.tmp.npy"
        np.save(tmp, predict(model, np.load(CACHE / "images" / f"{s}.npy"), tta=True, device=device))
        tmp.rename(prob_dir / f"{s}.npy")
        if i % 50 == 0:
            print(f"  probabilidades {i}/{len(todo)}", flush=True)
    del model
    empty_cache(device)


def build_dataset(run, stems, prob_dir, cfg, cache, workers, rebuild, members=None, mode="features",
                  direct_members=None, direct_confidence=False):
    """members: runs de los miembros del ensemble cuyas probabilidades (mismo subdirectorio que prob_dir) alimentan
    los rasgos de desacuerdo; None reproduce el conjunto de rasgos original."""
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)
    index = json.loads((CACHE / "index.json").read_text())
    jobs = [(s, index["annotations"][s], prob_dir / f"{s}.npy", cfg,
             [m / prob_dir.name / f"{s}.npy" for m in members] if members else None, mode,
             [m / prob_dir.name / f"{s}.lab.npy" for m in direct_members] if direct_members else None,
             direct_confidence) for s in stems]
    required = [p for job in jobs for p in (job[2], *(job[4] or []), *(job[6] or []))]
    if direct_confidence:
        required += [Path(str(p).replace(".lab.npy", ".npy")) for job in jobs for p in (job[6] or [])]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} mapas de candidatos/probabilidades; primero: {missing[0]}")
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        res = pool.map(one_image, jobs, chunksize=1)
    df = pd.concat([r[0] for r in res if not r[0].empty], ignore_index=True)
    gts = {r[1]["stem"]: r[1]["n_gt"] for r in res}
    pd.to_pickle((df, gts), cache)
    return df, gts


def best_threshold(df, gts, scores, min_keeps=(0, 1)):
    best = (0, None, None, None)
    for min_keep in min_keeps:
        for t in np.arange(0.1, 0.8, 0.025):
            r = pq_from(df, gts, kept_by_threshold(df, scores, t, min_keep))
            if r[0] > best[0]:
                best = (r[0], t, min_keep, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--fit", choices=["val", "train"], default="val")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-predict", action="store_true", help="--fit train: exigir probabilidades guardadas de toda la selección")
    ap.add_argument("--train-limit", type=int, default=0, help="--fit train: número de imágenes (0 = todas)")
    ap.add_argument("--train-sampling", choices=["stratified", "chronological"], default="stratified")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--target", choices=["match", "expected_iou"], default="match")
    ap.add_argument("--min-keep", type=int, choices=[0, 1], default=1,
                    help="1 reproduce el mínimo de candidatos del envío actual")
    ap.add_argument("--output-dir", type=Path, help="directorio alternativo para no sobrescribir el reclasificador vigente")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto (solo para predecir train)")
    ap.add_argument("--train-probs-dir", type=Path, help="#14: probabilidades de train alternativas (p.ej. runs/<name>/train_probs_oof, fuera de muestra)")
    ap.add_argument("--union-candidates", action="store_true",
                    help="ensembles: añadir los candidatos de cada miembro que el promedio no genera (rasgo from_member); usar con --output-dir")
    ap.add_argument("--direct-candidates", action="append", default=[],
                    help="run con mapas <split>_probs/<stem>.lab.npy de instancias directas (p.ej. YOLO); repetir si hay varias fuentes")
    ap.add_argument("--direct-confidence", action="store_true",
                    help="con --direct-candidates, añadir la confianza del detector como rasgo (opt-in)")
    ap.add_argument("--member-features", action="store_true",
                    help="ensembles: añadir rasgos de desacuerdo entre miembros (runs/<name>/members.json); usar con --output-dir")
    args = ap.parse_args()
    run = ROOT / "runs" / args.name
    out = args.output_dir or run / "rescore"
    out.mkdir(parents=True, exist_ok=True)
    post = json.loads((run / "postprocess.json").read_text())
    cfg = PostCfg(**{k: post[k] for k in PostCfg.__dataclass_fields__})
    members = None
    mode = "union" if args.union_candidates or args.direct_candidates else "features"
    if args.direct_candidates and not args.output_dir:
        ap.error("--direct-candidates requiere --output-dir para conservar el reclasificador vigente")
    if args.direct_confidence and not args.direct_candidates:
        ap.error("--direct-confidence requiere --direct-candidates")
    if args.member_features and args.union_candidates:
        ap.error("--member-features y --union-candidates son excluyentes")
    if args.member_features and args.direct_candidates:
        ap.error("--member-features y --direct-candidates son excluyentes")
    if args.member_features or args.union_candidates:
        member_names = json.loads((run / "members.json").read_text())
        members = [ROOT / "runs" / m for m in member_names]
        print(f"{'candidatos de la unión de' if mode == 'union' else 'rasgos de desacuerdo entre'} {len(members)} miembros: {member_names}")
    direct_members = [ROOT / "runs" / name for name in args.direct_candidates]

    df, gts = build_dataset(run, json.loads((CACHE / "index.json").read_text())["val"], run / "val_probs", cfg, out / "dataset.pkl", args.workers, args.rebuild, members, mode, direct_members, args.direct_confidence)
    feats = [c for c in df.columns if c not in ("stem", "id", "match", "y")]
    print(f"val: {len(df)} candidatos en {df.stem.nunique()} imágenes; {100 * (df.y > 0).mean():.1f}% casan con algún anotador")

    # comprobación: el filtro fijo del post-proceso ajustado sobre estos candidatos debe reproducir su PQ
    rule = ((df.area >= cfg.min_area) & (df.p_mean >= cfg.min_score)).values
    if mode == "union":
        rule = rule & (df.from_member == 0).values  # el filtro fijo solo conoce los candidatos del ensemble
        print(f"candidatos añadidos por los miembros: {int(df.from_member.sum())} de {len(df)} "
              f"({100 * (df[df.from_member == 1].y > 0).mean():.1f}% casan)")
    base = pq_from(df, gts, rule)
    print(f"filtro fijo {cfg}: PQ {base[0]:.4f} (postprocess.json: {post['pq_val']:.4f})")

    def target(frame):
        return frame.y.values if args.target == "match" else np.array([np.mean(m) for m in frame.match])

    X, y, groups = df[feats].values, target(df), df.stem.values
    if args.fit == "train":
        index = json.loads((CACHE / "index.json").read_text())
        index["train"] = select_observations(index["train"], args.train_limit, args.sample_seed, args.train_sampling)
        (out / "train_selection.json").write_text(json.dumps({"strategy": args.train_sampling,
            "seed": args.sample_seed, "stems": index["train"]}, indent=2) + "\n")
        train_probs = args.train_probs_dir or run / "train_probs"
        if args.no_predict:
            missing = [s for s in index["train"] if not (train_probs / f"{s}.npy").exists()]
            if missing:
                raise ValueError(f"Faltan probabilidades de {len(missing)} imágenes seleccionadas; "
                                 "quitar --no-predict. No se reduce la muestra silenciosamente.")
        else:
            print("prediciendo train (GPU)...", flush=True)
            ensure_probs(run, index["train"], train_probs, args.device)
        dt, _ = build_dataset(run, index["train"], train_probs, cfg, out / "train_dataset.pkl", args.workers, True, members, mode, direct_members, args.direct_confidence)
        print(f"train: {len(dt)} candidatos en {dt.stem.nunique()} imágenes; {100 * (dt.y > 0).mean():.1f}% casan"
              f"  (p_mean medio train {dt.p_mean.mean():.3f} vs val {df.p_mean.mean():.3f})")
        Xt, yt = dt[feats].values, target(dt)

    # C: solo val (validación cruzada)   A: solo train   B: train + folds de val
    variants = {"solo val (CV)": "val"} if args.fit == "val" else {
        "solo val (CV)": "val", "A: solo train": "train", "B: train + val (CV)": "both"}
    results = {}
    for label, kind in variants.items():
        oof = np.zeros(len(df))
        if kind == "train":
            oof = make_model().fit(Xt, yt).predict(X)
        else:
            for rep in range(args.repeats):
                for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, y, groups):
                    Xf, yf = (X[tr], y[tr]) if kind == "val" else (np.vstack([Xt, X[tr]]), np.concatenate([yt, y[tr]]))
                    oof[te] += make_model(rep).fit(Xf, yf).predict(X[te]) / args.repeats
        pq_b, t_b, mk_b, (_, tp, fp, fn) = best_threshold(df, gts, oof, min_keeps=(args.min_keep,))
        results[label] = (pq_b, t_b, mk_b, kind)
        print(f"{label:22s} PQ {pq_b:.4f}  t={t_b:.3f} min_keep={mk_b}  tp {tp} fp {fp} fn {fn}", flush=True)
    print(f"{'filtro fijo':22s} PQ {base[0]:.4f}  tp {base[1]} fp {base[2]} fn {base[3]}")

    label, (pq_b, t_b, mk_b, kind) = max(results.items(), key=lambda kv: kv[1][0])
    Xf, yf = {"val": (X, y), "train": (Xt, yt), "both": (np.vstack([Xt, X]), np.concatenate([yt, y]))
              }[kind] if args.fit == "train" else (X, y)
    final = make_model().fit(Xf, yf)
    joblib.dump({"model": final, "features": feats, "threshold": float(t_b), "min_keep": int(mk_b), "variant": label,
                 "cfg": cfg.as_dict(), "pq_oof": float(pq_b), "pq_rule": float(base[0]),
                 "target": args.target, "train_sampling": args.train_sampling,
                 "sample_seed": args.sample_seed, "train_probs_dir": str(train_probs) if args.fit == "train" else None,
                 "member_features": bool(members) and mode == "features", "union_candidates": mode == "union",
                 "members": member_names if members else None, "direct_candidates": args.direct_candidates,
                 "direct_confidence": args.direct_confidence}, out / "model.joblib")
    print(f"guardado «{label}» en {out / 'model.joblib'}")


if __name__ == "__main__":
    main()
