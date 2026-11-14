"""Ajusta los umbrales del post-proceso maximizando el PQ de validación.

1. Predice una vez las probabilidades de val (con TTA) y las guarda en runs/<name>/val_probs/.
2. Búsqueda en rejilla sobre PostCfg, paralela por imagen. El resultado de cada imagen se
   guarda en runs/<name>/tune_parts/ en cuanto termina: si se interrumpe, se reanuda.
3. Guarda el mejor en runs/<name>/postprocess.json.

Uso: ~/venvs/ml/bin/python scripts/tune_postprocess.py --name r34-base
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filaments.device import empty_cache, pick  # noqa: E402
from filaments.data import CACHE  # noqa: E402
from filaments.postprocess import PostCfg, instances, pq, pq_counts  # noqa: E402

GRID = {
    "t_fg": [0.4, 0.5, 0.6],
    "t_b": [0.5],
    "split": [True],
    "merge_dist": [8, 16, 24],
    "min_area": [300, 400, 600],
    "min_score": [0.7, 0.8, 0.9],
}

def _score_image(args):
    """Carga solo la imagen que evalúa (~0.75 GB de pico por proceso) y escribe su resultado a disco."""
    prob_path, ann_ids, cfgs, out_path = args
    prob = np.load(prob_path)
    gts = [np.load(CACHE / "labels" / f"{a}.npy") for a in ann_ids]
    out = []
    for cfg in cfgs:
        lab = instances(prob, cfg)
        out.append([pq_counts(g, lab) for g in gts])
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(out_path)
    return out_path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--reuse-probs", action="store_true", help="no volver a predecir si val_probs/ ya está completo")
    ap.add_argument("--workers", type=int, default=6, help="cada proceso usa ~0.5 GB de pico")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    args = ap.parse_args()
    device = pick(args.device)
    run = ROOT / "runs" / args.name
    cfg = json.loads((run / "config.json").read_text())
    index = json.loads((CACHE / "index.json").read_text())
    stems = index["val"]

    prob_dir = run / "val_probs"
    prob_dir.mkdir(exist_ok=True)
    if not (args.reuse_probs and all((prob_dir / f"{s}.npy").exists() for s in stems)):
        # torch solo aquí: los workers no lo importan (~0.4 GB menos cada uno)
        import torch

        from filaments.infer import predict
        from filaments.model import build

        model = build(cfg, pretrained=False).to(device).eval()
        model.load_state_dict(torch.load(run / args.ckpt, map_location=device))
        for s in stems:
            np.save(prob_dir / f"{s}.npy", predict(model, np.load(CACHE / "images" / f"{s}.npy"), tta=not args.no_tta, device=device))
        del model
        empty_cache(device)

    cfgs = [PostCfg(**dict(zip(GRID, v))) for v in itertools.product(*GRID.values())]
    # resultados parciales por imagen, ligados a la rejilla y a las probabilidades usadas
    tag = hashlib.md5(json.dumps([GRID, args.ckpt, args.no_tta]).encode()).hexdigest()[:8]
    part_dir = run / "tune_parts" / tag
    part_dir.mkdir(parents=True, exist_ok=True)
    # todas las configuraciones de una imagen en el mismo worker: las probabilidades se cargan una vez
    tasks = [(prob_dir / f"{s}.npy", index["annotations"][s], cfgs, part_dir / f"{s}.json")
             for s in stems if not (part_dir / f"{s}.json").exists()]
    print(f"{len(stems) - len(tasks)}/{len(stems)} imágenes ya evaluadas en {part_dir}", flush=True)
    # spawn: un fork del proceso con contexto CUDA hereda su memoria y reventó la RAM (OOM con 24 GB).
    # multiprocessing.Pool y no ProcessPoolExecutor: con max_tasks_per_child se cuelga en Python 3.12
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for i, stem in enumerate(pool.imap_unordered(_score_image, tasks), start=1):
            print(f"[{i}/{len(tasks)}] {stem}", flush=True)
    per_image = [json.loads((part_dir / f"{s}.json").read_text()) for s in stems]
    results = []
    for k, c in enumerate(cfgs):
        counts = [x for img in per_image for x in img[k]]
        results.append((pq(counts), c, sum(x[1] for x in counts), sum(x[2] for x in counts), sum(x[3] for x in counts)))
    results.sort(key=lambda r: -r[0])
    for r in results[:10]:
        print(f"PQ {r[0]:.4f}  tp {r[2]} fp {r[3]} fn {r[4]}  {r[1]}")
    reference = next((r for r in results if r[1] == PostCfg(min_area=100)), None)
    if reference:
        print(f"referencia (t_fg 0.5, split, sin merge, min_area 100): PQ {reference[0]:.4f}")
    best = results[0]
    (run / "postprocess.json").write_text(json.dumps({"pq_val": best[0], "tta": not args.no_tta, **best[1].as_dict()}, indent=2))


if __name__ == "__main__":
    main()
