"""Build a readable, self-contained inference notebook from the release sources.

This is a maintainer tool: the generated notebook never reads/imports these
Python files. Function bodies are copied verbatim, with local imports removed.
"""
import argparse
import ast
from pathlib import Path
import textwrap

import nbformat as nb


def definitions(root, relative, names):
    source = (root / relative).read_text()
    lines = source.splitlines(keepends=True)
    nodes = {node.name: node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    result = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        body = ''.join(lines[start:node.end_lineno])
        # The referenced functions are also present as notebook definitions.
        body = '\n'.join(line for line in body.splitlines()
                         if not line.lstrip().startswith(('from .', 'from filaments.')))
        result.append(body)
    return '\n\n\n'.join(result)


def build_notebook(root):
    cells = []

    def markdown(source):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source, tag='definitions'):
        cells.append(nb.v4.new_code_cell(textwrap.dedent(source).strip(), metadata={'tags': [tag]}))

    def include(relative, names):
        code(definitions(root, relative, names))

    markdown('''
        # Solar filament instance segmentation — complete reproduction code
        Shengkai Zhu.

        All project code needed to reproduce submissions **56835266** (match)
        and **56835269** (expected IoU) is included in the cells below. No cell
        invokes a Python script, imports `filaments`, writes a Python module, or
        loads source code from another file. The notebook uses the supplied
        weights, model configurations and fitted rescorers, plus standard libraries.

        This is the complete **inference and verification** pipeline for the
        trained ens5c ensemble. Training is not rerun. Install `requirements.txt`
        in a fresh Python 3.12 environment and open this notebook from the repository
        root. Supply the official competition data after accepting its terms.
        No test annotations are read. Reference predictions are opened only after
        fresh inference, to measure reproduction agreement, not test accuracy.
    ''')
    markdown('''
        ## 1. Configuration and dependencies
        `DEVICE="cuda"` and `LIMIT=0` reproduce the full 180-image test set.
        For a CPU smoke run use `DEVICE="cpu"` and `LIMIT=1`; partial outputs
        have a `smoke_` prefix and must not be submitted. Set `FILAMENT_DATA_ROOT`
        to the directory containing `train/` and `test/`, or edit `DATA` below.
        The optional training cache is unnecessary for inference and is disabled
        by default. Use one GPU job at a time and allow at least 6 GB free RAM.
    ''')
    code('''
        from pathlib import Path
        import os
        import json
        import hashlib
        import contextlib
        from collections import defaultdict
        from dataclasses import dataclass, field, asdict

        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['OPENBLAS_NUM_THREADS'] = '2'

        import cv2
        import joblib
        import numpy as np
        import pandas as pd
        import torch
        from torch import nn
        import segmentation_models_pytorch as smp
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed
        from pycocotools import mask as mask_util

        ROOT = Path.cwd().resolve()
        assert (ROOT / 'weights/manifest.json').is_file(), 'Run from repository root'
        DEVICE = os.environ.get('FILAMENT_DEVICE', 'cuda')
        LIMIT = int(os.environ.get('FILAMENT_LIMIT', '0'))
        assert 0 <= LIMIT <= 180
        BUILD_TRAINING_CACHE = False
        OUTPUTS = Path(os.environ.get('FILAMENT_OUTPUT_DIR', str(ROOT / 'outputs')))
        DATA = Path(os.environ.get('FILAMENT_DATA_ROOT',
                                  str(ROOT / 'data/MAGFiLO_1.0_Kaggle_2026'))).resolve()
        if not DATA.exists() and 'FILAMENT_DATA_ROOT' not in os.environ:
            matches = list(Path('/kaggle/input').rglob(
                'MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
            assert len(matches) == 1, 'Set FILAMENT_DATA_ROOT to the official dataset'
            DATA = matches[0].parent.parent
        TRAIN_JSON = DATA / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
        TRAIN_IMAGES = DATA / 'train/train_images'
        TEST_IMAGES = DATA / 'test/test_images'
        CACHE = ROOT / 'data/cache'
        assert TRAIN_JSON.is_file() and TEST_IMAGES.is_dir(), DATA
        MEMBERS = ['r50-long', 'cnx-s-bce', 'cnx-s-long', 'r50-consensus', 'cnx-s-consensus']
        SIZE = 2048
        MEAN, STD = 0.45, 0.225
        FLIPS = [(), (-1,), (-2,), (-2, -1)]
        OBSERVATORIES = 'BCLMTU'
        MIN_CANDIDATE = 50
        torch.set_num_threads(2)
        print('Data:', DATA, '| Device:', DEVICE, '| Images:', LIMIT or 180)
    ''', 'setup')
    markdown('''
        ## 2. Reconstruct and verify the five checkpoints
        The chunks contain the exact fine-tuned parameters. Verify chunk and
        checkpoint SHA-256 hashes; no external pretrained weights are downloaded.
    ''')
    include('scripts/restore_weights.py', ['sha'])
    restoration = definitions(root, 'scripts/restore_weights.py', ['main'])
    code(restoration.replace('def main():', 'def restore_weights():', 1))
    code('restore_weights()', 'restore')
    markdown('''
        ## 3. Images, COCO masks and observation split
        Images remain grayscale at 2048 × 2048. Split by observation, so all
        annotations of the same image stay together. The original 601/106 split
        and all 180 test IDs are verified before inference.
    ''')
    include('src/filaments/rle.py', ['encode_counts', 'decode_counts', 'polygon_to_mask'])
    include('src/filaments/data.py', ['Observation', 'load_observations', 'split', 'rasterize', 'read_image'])
    include('scripts/prepare_cache.py', ['cache_image', 'cache_label'])
    code('''
        observations = load_observations()
        train_stems, val_stems = split(list(observations))
        test_paths = sorted(TEST_IMAGES.glob('*.jpeg'))
        original = json.loads((ROOT / 'metadata/original_split.json').read_text())
        generated = dict(train=train_stems, val=val_stems, test=[p.stem for p in test_paths])
        assert all(generated[key] == original[key] for key in ('train', 'val', 'test'))
        assert len(test_paths) == 180
        assert not set(train_stems) & set(val_stems)
        if BUILD_TRAINING_CACHE:
            (CACHE / 'images').mkdir(parents=True, exist_ok=True)
            (CACHE / 'labels').mkdir(parents=True, exist_ok=True)
            for path in sorted(TRAIN_IMAGES.glob('*.jpeg')) + test_paths:
                cache_image(path)
            items = [(ann, polygons) for obs in observations.values()
                     for ann, polygons in obs.annotations.items()]
            generated['centroids'] = dict(cache_label(item) for item in items)
            generated['annotations'] = {s: sorted(o.annotations) for s, o in observations.items()}
            (CACHE / 'index.json').write_text(json.dumps(generated))
        print('Verified split:', len(train_stems), len(val_stems), len(test_paths))
    ''', 'data')
    markdown('''
        ## 4. U-Net and four-flip test-time augmentation
        Two ResNet50 and three ConvNeXt-S encoders produce foreground and contact
        logits. The exact architecture comes from each supplied configuration.
        Each member averages four flips; probabilities are converted to float16
        as in the accepted pipeline, then float32 for the five-member average.
    ''')
    include('src/filaments/device.py', ['pick', 'autocast'])
    include('src/filaments/model.py', ['FilamentNet', 'build'])
    include('src/filaments/infer.py', ['predict', 'predict_tiled'])
    markdown('''
        ## 5. Contact-aware watershed and fragment merging
        Threshold the foreground, seed away from contact regions, apply watershed,
        and merge nearby fragments according to the fitted configuration.
    ''')
    include('src/filaments/postprocess.py', ['PostCfg', 'instances', '_merge_fragments', '_filter'])
    markdown('''
        ## 6. Candidate features and instance rescoring
        Compute morphology, probability, contrast, disk position, year and
        observatory features. The supplied histogram gradient-boosting models
        score candidates. Keep candidates above the fitted threshold, retaining
        at least the highest-scoring candidate when any exist.
    ''')
    include('src/filaments/rescore.py', ['candidates', 'disk', 'features', 'keep'])
    markdown('''
        ## 7. CSV encoding and structural audit
        Encode disjoint masks as compressed COCO RLE. Check column names,
        image coverage, unique instance IDs, nonempty masks and zero overlaps.
    ''')
    include('src/filaments/submission.py', ['build_submission', 'validate'])
    include('scripts/final_submission_audit.py', ['audit'])
    markdown('''
        ## 8. Load weights and run fresh inference
        Both rescorers share the same five-model inference pass. Every image is
        inferred afresh: the notebook does not reuse cached predictions. It writes
        outputs only after model inference, then compares with the accepted CSVs.
        CPU inference can take several minutes per image.
    ''')
    # Retain the accepted orchestration verbatim except for notebook settings,
    # removal of the script-source cache, and a returned report for the next cell.
    source = (root / 'scripts/reproduce_predictions.py').read_text()
    start = source.index('    torch.set_num_threads(2)')
    end = source.index('\n\nif __name__')
    body = source[start:end]
    body = body.replace('args.device', 'device').replace('args.limit', 'limit')
    body = body.replace('    outputs = ROOT / "outputs"', '    outputs = OUTPUTS')
    body = body.replace('outputs.mkdir(exist_ok=True)', 'outputs.mkdir(parents=True, exist_ok=True)')
    first = body.index('    # Immutable package inputs')
    last = body.index('    for n, path in enumerate(paths, 1):')
    body = body[:first] + body[last:]
    first = body.index('        stored = cache /')
    last = body.index('        for variant in judges:', first)
    computation = body[body.index('            image = read_image(path)', first):
                       body.index('            temporary = stored.with_suffix', first)]
    computation = '\n'.join(line[4:] for line in computation.splitlines()) + '\n'
    body = body[:first] + computation + body[last:]
    body = body.replace('coco.iou(', 'mask_util.iou(')
    body += '\n    return report\n'
    code('def reproduce_predictions(device=DEVICE, limit=LIMIT):\n' + body)
    code('reproduction_report = reproduce_predictions()', 'inference')
    markdown('''
        ## 9. Audit results
        For a full run, every official test image must be represented. For a smoke
        run, require exactly the selected subset and label the audit as partial.
        Prediction-agreement PQ compares two prediction files; it is **not**
        accuracy against the hidden test labels. Hardware/library differences
        can affect contour pixels. The original 180-image CUDA reproduction is
        recorded separately in `metadata/full_reproduction.json`.
    ''')
    code('''
        audit_results = {}
        for variant in ('match', 'eiou'):
            prefix = 'smoke_' if LIMIT else ''
            target = OUTPUTS / f'{prefix}submission_ens5c_rescore_{variant}.csv'
            if LIMIT:
                # The original audit expects a directory of expected image IDs.
                # Temporary empty files specify the smoke subset, not image data.
                import tempfile
                with tempfile.TemporaryDirectory() as temporary:
                    subset = Path(temporary)
                    for path in test_paths[:LIMIT]:
                        (subset / path.name).touch()
                    result = audit(target, subset)
            else:
                result = audit(target, TEST_IMAGES)
            result['full_dataset'] = not bool(LIMIT)
            (OUTPUTS / f'{prefix}{variant}_audit.json').write_text(json.dumps(result, indent=2))
            assert result['passed'], result['errors']
            audit_results[variant] = result
        print(json.dumps({'reproduction': reproduction_report, 'audits': audit_results}, indent=2))
    ''', 'audit')
    markdown('''
        ## Evaluation and training evidence
        The report describes the training procedure, annotation disagreement,
        morphology and limitations. `runs/*/config.json` and `log.csv` preserve
        training settings and histories. Training and fitting from scratch remain
        available as separate research scripts; they are not dependencies of this
        notebook. Local nested rescorer validation reused a development split
        previously used to select segmenters and post-processing, so it is not
        an independent estimate of the complete model-selection process.
    ''')
    for index, cell in enumerate(cells):
        cell.id = f'ens5c-{index:02d}'
    notebook = nb.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    notebook.metadata.language_info = dict(name='python', version='3.12')
    nb.validate(notebook)
    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    destination = args.root / 'reproduce.ipynb'
    nb.write(build_notebook(args.root), destination)
    print(destination)
