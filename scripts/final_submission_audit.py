"""Audit submission IDs, RLE, coverage and pairwise non-overlap per observation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def audit(path, test_dir):
    import numpy as np
    import pandas as pd
    from filaments.submission import validate
    from filaments.rle import decode_counts
    frame = pd.read_csv(path)
    errors = validate(frame, test_dir)
    stems = frame.filament_id.str.rsplit("_", n=1).str[0]
    overlap_pixels = 0
    for stem, group in frame.groupby(stems):
        occupied = np.zeros((2048, 2048), bool)
        for encoded in group.segmentation_rle:
            mask = decode_counts(encoded).astype(bool)
            overlap_pixels += int((occupied & mask).sum())
            occupied |= mask
    if overlap_pixels:
        errors.append(f"Overlapping masks: {overlap_pixels} repeated pixels")
    return dict(csv=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                rows=len(frame), observations=stems.nunique(), overlapping_pixels=overlap_pixels,
                errors=errors, passed=not errors)


def main():
    from filaments.data import TEST_IMAGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=TEST_IMAGES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.csv, args.test_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
