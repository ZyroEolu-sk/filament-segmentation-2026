"""Crea un «run» de ensemble promediando las probabilidades guardadas de varios runs.

runs/<out>/{config.json, val_probs/, train_probs/}: val_probs con las imágenes comunes de val (deben ser las
106); train_probs con las imágenes de train que tengan probabilidades en TODOS los miembros. Después:
  tune_postprocess.py --name <out> --reuse-probs
  rescore_train.py --name <out> --fit train --train-limit N --no-predict
  predict_test.py --name m1 --name m2 --post-from <out> --rescore

Uso: ~/venvs/ml/bin/python scripts/make_ensemble.py --out ens-r50 --member r50-base --member r50-long
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--weight", action="append", type=float, help="peso por miembro (mismo orden; por defecto iguales)")
    args = ap.parse_args()
    weights = args.weight or [1.0] * len(args.member)
    if len(weights) != len(args.member):
        ap.error("--weight debe darse una vez por miembro")
    weights = np.array(weights, np.float32) / sum(weights)
    members = [ROOT / "runs" / m for m in args.member]
    out = ROOT / "runs" / args.out
    out.mkdir(exist_ok=True)
    shutil.copy(members[0] / "config.json", out / "config.json")
    (out / "members.json").write_text(json.dumps(args.member))
    if args.weight:
        (out / "weights.json").write_text(json.dumps(dict(zip(args.member, weights.tolist()))))
    for sub in ["val_probs", "train_probs"]:
        stems = set.intersection(*[{p.stem for p in (m / sub).glob("*.npy") if not p.name.endswith(".tmp.npy")} for m in members])
        (out / sub).mkdir(exist_ok=True)
        done = 0
        for s in sorted(stems):
            dst = out / sub / f"{s}.npy"
            if dst.exists():
                continue
            np.save(dst, sum(w * np.load(m / sub / f"{s}.npy").astype(np.float32) for w, m in zip(weights, members)).astype(np.float16))
            done += 1
        print(f"{sub}: {len(stems)} imágenes comunes, {done} nuevas")


if __name__ == "__main__":
    main()
