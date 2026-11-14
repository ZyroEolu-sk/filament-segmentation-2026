"""Construye data/cache: imágenes y mapas de etiquetas en .npy + index.json con el split.

Uso: ~/venvs/ml/bin/python scripts/prepare_cache.py
"""

import json
import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filaments.data import CACHE, TEST_IMAGES, TRAIN_IMAGES, load_observations, rasterize, read_image, split  # noqa: E402


def cache_image(path: Path) -> None:
    out = CACHE / "images" / f"{path.stem}.npy"
    if not out.exists():
        np.save(out, read_image(path))


def cache_label(item: tuple[str, list]) -> tuple[str, list]:
    ann_id, polygons = item
    out = CACHE / "labels" / f"{ann_id}.npy"
    lab = rasterize(polygons)
    if not out.exists():
        np.save(out, lab)
    cents = []
    for k in range(1, len(polygons) + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys):
            cents.append([float(ys.mean()), float(xs.mean())])
    return ann_id, cents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--spawn", action="store_true", help="opt-in: explicit spawn workers")
    args = parser.parse_args()
    (CACHE / "images").mkdir(parents=True, exist_ok=True)
    (CACHE / "labels").mkdir(parents=True, exist_ok=True)
    obs = load_observations()
    train, val = split(list(obs))
    paths = sorted(TRAIN_IMAGES.glob("*.jpeg")) + sorted(TEST_IMAGES.glob("*.jpeg"))
    items = [(a, p) for o in obs.values() for a, p in o.annotations.items()]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn") if args.spawn else None) as ex:
        list(ex.map(cache_image, paths, chunksize=8))
        centroids = dict(ex.map(cache_label, items, chunksize=8))
    index = {
        "train": train,
        "val": val,
        "test": sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg")),
        "annotations": {s: sorted(o.annotations) for s, o in obs.items()},
        "centroids": centroids,
    }
    (CACHE / "index.json").write_text(json.dumps(index))
    print(f"train {len(train)} obs · val {len(val)} obs · test {len(index['test'])} · anotaciones {len(items)}")


if __name__ == "__main__":
    main()
