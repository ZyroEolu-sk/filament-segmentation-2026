"""Predice (con TTA) una lista de imágenes con el best.pt de un run y guarda las probabilidades en un directorio.

Uso #14: tras entrenar `r50-consensus-f0..f4` con --fold, predecir las imágenes excluidas de cada fold y reunirlas en
`runs/r50-consensus/train_probs_oof/`, que luego usa `rescore_train.py --train-probs-dir`.

Uso: ~/venvs/ml/bin/python scripts/predict_stems.py --name r50-consensus-f0 --stems runs/r50-consensus-f0/heldout.json \
        --out runs/r50-consensus/train_probs_oof [--device mps]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rescore_train import ensure_probs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stems", type=Path, required=True, help="JSON con la lista de stems")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    stems = json.loads(args.stems.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_probs(ROOT / "runs" / args.name, stems, args.out, args.device)
    done = sum((args.out / f"{s}.npy").exists() for s in stems)
    print(f"{done}/{len(stems)} probabilidades en {args.out}")


if __name__ == "__main__":
    main()
