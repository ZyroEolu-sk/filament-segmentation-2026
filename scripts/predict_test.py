"""Genera outputs/submission_<name>.csv para el test con el modelo y el post-proceso ajustado.

Uso: ~/venvs/ml/bin/python scripts/predict_test.py --name r34-base [--name otro ...]
Con varios --name hace ensemble promediando probabilidades (usa el post-proceso del primero).
Con --rescore, el filtro fijo de área/confianza se sustituye por el reclasificador de instancias
entrenado con scripts/rescore_train.py (runs/<primer name>/rescore/model.joblib).
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filaments.device import empty_cache, pick  # noqa: E402
from filaments.data import CACHE, TEST_IMAGES  # noqa: E402
from filaments.infer import predict  # noqa: E402
from filaments.model import build  # noqa: E402
from filaments.postprocess import PostCfg, instances  # noqa: E402
from filaments.rescore import candidates, features, keep, union_candidates, with_direct_confidence, with_member_features  # noqa: E402
from filaments.submission import build_submission, write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--rescore", action="store_true", help="usar el reclasificador de instancias en vez del filtro fijo")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--tta8", action="store_true", help="opt-in: cuatro flips adicionales tras rotar 90 grados")
    ap.add_argument("--post-from", help="run del que tomar postprocess.json y rescore/ (p.ej. un ensemble de make_ensemble.py)")
    ap.add_argument("--rescore-path", type=Path, help="modelo alternativo, p.ej. runs/ens5c/rescore_yolo/model.joblib")
    ap.add_argument("--output", type=Path, help="CSV de salida; obligatorio con --rescore-path")
    args = ap.parse_args()
    if args.rescore_path and (not args.rescore or not args.output):
        ap.error("--rescore-path requiere --rescore y --output para no sobrescribir el CSV habitual")
    runs = [ROOT / "runs" / n for n in args.name]
    device = pick(args.device)
    post_run = ROOT / "runs" / args.post_from if args.post_from else runs[0]
    post = json.loads((post_run / "postprocess.json").read_text())
    tta = post.pop("tta")
    if args.tta8 and not tta:
        ap.error("--tta8 requiere TTA de cuatro flips en postprocess.json")
    post.pop("pq_val")
    post = PostCfg(**post)
    rescorer = joblib.load(args.rescore_path or post_run / "rescore" / "model.joblib") if args.rescore else None
    member_feats = bool(rescorer and rescorer.get("member_features"))
    union = bool(rescorer and rescorer.get("union_candidates"))
    if (member_feats or union) and rescorer.get("members") and rescorer["members"] != args.name:
        raise SystemExit(f"el reclasificador espera los miembros {rescorer['members']} en ese orden; --name da {args.name}")
    if rescorer and rescorer.get("direct_candidates"):
        if not args.post_from:
            raise SystemExit("el reclasificador con detector requiere --post-from del ensemble original")
        members_file = post_run / "members.json"
        expected = json.loads(members_file.read_text()) if members_file.exists() else [post_run.name]
        if expected != args.name:
            raise SystemExit(f"el ensemble espera los miembros {expected} en ese orden; --name da {args.name}")
    models = []
    for run in runs:
        m = build(json.loads((run / "config.json").read_text()), pretrained=False).to(device).eval()
        m.load_state_dict(torch.load(run / args.ckpt, map_location=device))
        models.append(m)

    stems = sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg"))
    parts = []
    for s in stems:
        img = np.load(CACHE / "images" / f"{s}.npy")
        probs = []
        for m in models:
            p4 = predict(m, img, tta=tta, device=device).astype(np.float32)
            if args.tta8:
                rotated = predict(m, np.rot90(img, 1), tta=True, device=device)
                p4 = (p4 + np.rot90(rotated, -1, axes=(-2, -1)).astype(np.float32)) / 2
            probs.append(p4)
        prob = np.mean(probs, axis=0)
        if rescorer:
            if union:
                direct_labs = []
                direct_probs = []
                for name in rescorer.get("direct_candidates", []):
                    path = ROOT / "runs" / name / "test_probs" / f"{s}.lab.npy"
                    if not path.exists():
                        raise FileNotFoundError(f"Falta {path}; exportar el test del detector antes de predecir")
                    direct_labs.append(np.load(path))
                    if rescorer.get("direct_confidence"):
                        direct_probs.append(np.load(ROOT / "runs" / name / "test_probs" / f"{s}.npy", mmap_mode="r"))
                lab, origin = union_candidates(prob, probs if rescorer.get("members") else [],
                                               PostCfg(**rescorer["cfg"]), direct_labs=direct_labs)
                feats = features(prob, img, lab, s)
                if len(feats):
                    feats["from_member"] = (origin[feats.id.values - 1] > 0).astype(float)
                if direct_probs:
                    feats = with_direct_confidence(feats, direct_probs, lab)
            else:
                lab = candidates(prob, PostCfg(**rescorer["cfg"]))
                feats = features(prob, img, lab, s)
            if member_feats:
                feats = with_member_features(feats, probs, lab, rescorer["cfg"]["t_fg"])
            scores = rescorer["model"].predict(feats[rescorer["features"]].values) if len(feats) else np.zeros(0)
            lab = keep(lab, scores, rescorer["threshold"], min_keep=1)  # el CSV necesita al menos una fila por imagen
        else:
            lab = instances(prob, post)
        if lab.max() == 0:
            # el CSV necesita al menos una fila por imagen: la instancia más grande con filtro relajado
            loose = instances(prob, PostCfg(t_fg=post.t_fg, t_b=post.t_b, split=post.split, merge_dist=post.merge_dist, min_area=50))
            if loose.max() > 0:
                lab = (loose == np.bincount(loose.ravel())[1:].argmax() + 1).astype(np.int32)
            print(f"{s}: sin instancias tras el filtro, fallback con {int(lab.max())}", flush=True)
        # RLE por imagen: guardar las máscaras 2048x2048 de todo el test son decenas de GB de RAM
        parts.append(build_submission({s: [lab == k for k in range(1, lab.max() + 1)]}))
    df = pd.concat(parts, ignore_index=True)
    out = args.output or ROOT / "outputs" / f"submission_{'+'.join(args.name)}{'_rescore' if args.rescore else ''}{'_tta8' if args.tta8 else ''}.csv"
    out.parent.mkdir(exist_ok=True)
    write(df, out, TEST_IMAGES)
    print(f"{out}: {len(df)} filamentos en {df.filament_id.str.rsplit('_', n=1).str[0].nunique()} imágenes")


if __name__ == "__main__":
    main()
