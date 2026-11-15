# Solar Filament Segmentation Challenge 2026 — ens5c

Shengkai Zhu.

This repository reproduces two accepted competition submissions with public PQ **0.40**:

| Submission | Rescorer | Instances | Local nested-rescorer PQ |
|---|---|---:|---:|
| 56835266 | Match probability (primary) | 1,214 | 0.47404 |
| 56835269 | Expected matched IoU (alternative) | 1,221 | 0.47236 |

Both cover all 180 official test observations. Local scores are exploratory: the
segmenters, ensemble and post-processing were selected on the same development
split. Nested folds isolate rescorer fitting and threshold selection, not the
entire model-selection process. The two local scores are not conclusively different.

## Reproduce without training

Use Python 3.12, a CUDA-compatible PyTorch installation, an NVIDIA GPU with at
least 8 GB VRAM, and approximately 15 GB free disk space including the official
data cache. Install the pinned dependencies in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Obtain the official data from the [competition data page](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
after accepting its terms. Set `FILAMENT_DATA_ROOT` to the directory containing
`train/` and `test/`. Run all cells of **[reproduce.ipynb](reproduce.ipynb)** from
the repository root. Kaggle's competition input mount is detected automatically.
The notebook:

1. Reconstructs the exact five checkpoints from the included weight chunks and verifies SHA-256 hashes.
2. Creates image/label caches from the official data and checks the original 601/106 observation split.
3. Performs fresh image inference with the five trained models; both rescorers share that pass.
4. Writes both CSVs, checks nonempty RLE, image coverage, unique IDs and non-overlapping masks, and measures agreement with the accepted reference predictions.

The reference CSVs are read **only after inference**, for comparison. Test labels
are neither needed nor used. No external pretrained checkpoint is downloaded at
inference time. Weight chunks are ordinary repository files, not Git LFS pointers.

The generated files are:

```text
outputs/submission_ens5c_rescore_match.csv
outputs/submission_ens5c_rescore_eiou.csv
outputs/reproduction_report.json
outputs/match_audit.json
outputs/eiou_audit.json
```

Cross-device floating-point differences can affect contour pixels. The report
therefore records both exact-row equality and decoded prediction-agreement PQ.
Agreement with a previous prediction file is **not** test-set accuracy.

Full reproduction was verified on 14 November 2026: fresh CUDA inference over all
180 test images produced both accepted CSVs byte for byte. Evidence is in
`metadata/full_reproduction.json`.

For a CPU smoke test after restoring weights and placing the official data under
`data/MAGFiLO_1.0_Kaggle_2026/`:

```bash
python scripts/restore_weights.py
python scripts/reproduce_predictions.py --device cpu --limit 1
```

This produces explicitly named partial smoke outputs; it is not a full submission.

## Method

The five U-Nets use two ResNet50 encoders initialized from ImageNet and three
ConvNeXt-S encoders with DINOv3 initialization. Training uses native 512-pixel
grayscale crops, geometric/photometric augmentation, random-annotator or consensus
targets, foreground/contact heads, BCE + Dice losses, AdamW, a plateau learning-rate
schedule and EMA. Four-flip TTA probabilities are averaged equally across models.
Contact-aware watershed and fragment merging produce disjoint candidate instances.
A histogram gradient-boosting rescorer filters those candidates.

The primary pipeline, configurations, diagnostics and limitations are described
in `report/technical_report.pdf`. Configurations and original logs are under
`runs/<member>/`. `metadata/original_split.json` contains observation IDs only.
The official training annotations remain in the user-supplied dataset.

## Layout

```text
reproduce.ipynb          end-to-end inference notebook
requirements.txt        pinned environment
src/filaments/          data, models, inference, post-processing, metrics
scripts/                inference, reconstruction, audit and training entry points
weights/                exact checkpoint chunks and SHA-256 manifest
runs/                   configurations, logs, rescorers and evaluation evidence
reference_predictions/  accepted CSVs used only for final comparison
report/                 four-page report and source
metadata/               split IDs, release metadata and structural audits
third_party/            pretraining terms and attribution
```

## Data and attribution

Only permitted competition training observations provide task labels. The public
MAGFiLO annotation archive is not used because it contains competition test
observations. Competition images and annotation polygons are not redistributed.
The fine-tuned weights retain applicable pretraining-model terms; see
`third_party/DINOv3-LICENSE.md` and `third_party/ATTRIBUTION.md`.

This package describes the accepted **ens5c baseline**. Unfinished all-data
retraining, transformer trials, and detector/refiner experiments are not silently
substituted into the submitted model.
