"""Real image inference for both accepted ens5c submissions, without retraining."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MEMBERS = ["r50-long", "cnx-s-bce", "cnx-s-long", "r50-consensus", "cnx-s-consensus"]


def main():
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from pycocotools import mask as coco
    from filaments.data import CACHE, TEST_IMAGES, read_image
    from filaments.model import build
    from filaments.infer import predict
    from filaments.device import pick
    from filaments.postprocess import PostCfg, instances
    from filaments.rescore import candidates, features, keep
    from filaments.submission import build_submission

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="smoke test only; does not produce a full submission")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = pick(args.device)
    paths = sorted(TEST_IMAGES.glob("*.jpeg"))
    assert len(paths) == 180, "Expected 180 official test images"
    if args.limit:
        paths = paths[:args.limit]
    judges = {key:joblib.load(ROOT / "runs/ens5c" / sub / "model.joblib")
              for key,sub in [("match","rescore"),("eiou","rescore_expected_iou")]}
    models = []
    for member in MEMBERS:
        folder = ROOT / "runs" / member
        net = build(json.loads((folder / "config.json").read_text()), pretrained=False)
        net.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
        models.append(net.to(device).eval())
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    parts = {key:[] for key in judges}
    # Immutable package inputs define whether cached per-image rows are reusable.
    signature = hashlib.sha256((ROOT / "weights/manifest.json").read_bytes() +
        b"".join((ROOT / "runs/ens5c" / sub / "model.joblib").read_bytes()
                 for sub in ["rescore","rescore_expected_iou"]) + Path(__file__).read_bytes() +
        str((torch.__version__, device)).encode()).hexdigest()[:16]
    cache = outputs / "parts" / signature
    cache.mkdir(parents=True, exist_ok=True)
    for n, path in enumerate(paths, 1):
        stored = cache / f"{path.stem}.json"
        if stored.exists():
            rows = json.loads(stored.read_text())
        else:
            image = read_image(path)
            probs = [predict(model, image, tta=True, device=device).astype(np.float32) for model in models]
            probability = np.mean(probs, axis=0)
            rows = {}
            for variant, judge in judges.items():
                cfg = PostCfg(**judge["cfg"])
                labels = candidates(probability, cfg)
                feats = features(probability, image, labels, path.stem)
                scores = judge["model"].predict(feats[judge["features"]].values) if len(feats) else np.zeros(0)
                labels = keep(labels, scores, judge["threshold"], min_keep=1)
                if labels.max() == 0:
                    loose = instances(probability, PostCfg(t_fg=cfg.t_fg, t_b=cfg.t_b, split=cfg.split,
                                                          merge_dist=cfg.merge_dist, min_area=50))
                    if loose.max():
                        labels = (loose == np.bincount(loose.ravel())[1:].argmax()+1).astype(np.int32)
                frame = build_submission({path.stem:[labels == k for k in range(1, int(labels.max())+1)]})
                rows[variant] = frame.to_dict("records")
            temporary = stored.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows))
            temporary.replace(stored)
        for variant in judges:
            parts[variant].extend(rows[variant])
        print(f"{n}/{len(paths)} {path.stem}", flush=True)
    report = dict(full_dataset=not args.limit, images=len(paths), device=device, torch=torch.__version__,
                  checkpoint_manifest_sha256=hashlib.sha256((ROOT / "weights/manifest.json").read_bytes()).hexdigest(),
                  reference_used_for_inference=False, variants={})
    for variant in judges:
        filename = f"submission_ens5c_rescore_{variant}.csv"
        target = outputs / (f"smoke_{filename}" if args.limit else filename)
        frame = pd.DataFrame(parts[variant], columns=["filament_id","segmentation_rle"])
        frame.to_csv(target,index=False)
        reference = pd.read_csv(ROOT / "reference_predictions" / filename)
        allowed = {p.stem for p in paths}
        reference = reference[reference.filament_id.str.rsplit("_",n=1).str[0].isin(allowed)]
        a = frame.filament_id.str.rsplit("_",n=1).str[0]
        b = reference.filament_id.str.rsplit("_",n=1).str[0]
        total_iou = tp = fp = fn = 0
        for stem in sorted(allowed):
            masks_a = [dict(size=[2048,2048],counts=c.encode()) for c in frame.loc[a==stem,"segmentation_rle"]]
            masks_b = [dict(size=[2048,2048],counts=c.encode()) for c in reference.loc[b==stem,"segmentation_rle"]]
            if masks_a and masks_b:
                iou = coco.iou(masks_a,masks_b,[0]*len(masks_b))
                hit = iou > .5
                total_iou += float(iou[hit].sum()); tp += int(hit.sum())
                fp += int((hit.sum(1)==0).sum()); fn += int((hit.sum(0)==0).sum())
            else:
                fp += len(masks_a); fn += len(masks_b)
        report["variants"][variant] = dict(rows=len(frame), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            exact_rows=frame.reset_index(drop=True).equals(reference.reset_index(drop=True)),
            prediction_agreement_pq=total_iou/max(tp+.5*(fp+fn),1),
            agreement_note="Comparison with submitted predictions, not test accuracy")
    output = outputs / ("smoke_reproduction_report.json" if args.limit else "reproduction_report.json")
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
