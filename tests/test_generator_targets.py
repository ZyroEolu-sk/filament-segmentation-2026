"""merge_targets (consenso / mayoría) y el reparto de recortes del generador, sin datos del concurso."""

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import filaments.data as fdata  # noqa: E402
from filaments.data import CropGenerator, merge_targets  # noqa: E402


def _lab(*boxes, shape=(64, 64)):
    lab = np.zeros(shape, np.uint8)
    for k, (y0, y1, x0, x1) in enumerate(boxes, start=1):
        lab[y0:y1, x0:x1] = k
    return lab


class MergeTargetsTests(unittest.TestCase):
    def test_single_annotator_is_original_behaviour(self):
        lab = _lab((10, 20, 10, 40))
        for mode in ("random", "consensus", "majority"):
            t = merge_targets([lab], mode)
            self.assertEqual(t.shape, (3, 64, 64))
            np.testing.assert_array_equal(t[0], (lab > 0).astype(np.float32))

    def test_consensus_is_fraction_and_majority_is_half(self):
        a = _lab((10, 20, 10, 40))            # los tres lo marcan
        b = _lab((10, 20, 10, 40), (40, 44, 5, 30))  # el segundo filamento solo lo marca un anotador
        c = _lab((10, 20, 10, 40))
        t = merge_targets([a, b, c], "consensus")
        self.assertAlmostEqual(float(t[0][15, 20]), 1.0)
        self.assertAlmostEqual(float(t[0][42, 10]), 1 / 3, places=5)
        m = merge_targets([a, b, c], "majority")
        self.assertEqual(float(m[0][15, 20]), 1.0)
        self.assertEqual(float(m[0][42, 10]), 0.0)
        two = merge_targets([a, b], "majority")  # 1 de 2 = la mitad: se conserva
        self.assertEqual(float(two[0][42, 10]), 1.0)

    def test_contact_is_union_and_skeleton_follows_majority(self):
        a = _lab((10, 20, 10, 40), (21, 30, 10, 40))  # dos filamentos que se tocan en el anotador a
        b = _lab((10, 30, 10, 40))                     # uno solo en b
        t = merge_targets([a, b], "consensus")
        self.assertTrue(t[1].any(), "la zona de contacto de a debe conservarse")
        self.assertTrue(t[2][15, 20:30].any() or t[2][20:25, 20:30].any())
        self.assertTrue(set(np.unique(t[2])) <= {0.0, 1.0})

    def test_random_mode_rejects_several_annotators(self):
        with self.assertRaises(ValueError):
            merge_targets([_lab((0, 5, 0, 5)), _lab((0, 5, 0, 5))], "random")


class YearWeightTests(unittest.TestCase):
    """#7: sobremuestreo por año opt-in, con un índice sintético (sin datos del concurso)."""

    def _generator(self, **kw):
        stems = ["20110101000000Ah", "20120101000000Ah", "20210101000000Ah"]
        index = {"annotations": {s: [s + "_0"] for s in stems}, "centroids": {s + "_0": [] for s in stems}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fdata, "CACHE", Path(tmp)):
            (Path(tmp) / "index.json").write_text(json.dumps(index))
            return CropGenerator(stems, **kw)

    def test_default_keeps_uniform_sampling(self):
        self.assertIsNone(self._generator().stem_p)

    def test_weights_by_year(self):
        gen = self._generator(year_weights={"2021": 2.0})
        np.testing.assert_allclose(gen.stem_p, [0.25, 0.25, 0.5])

    def test_rejects_non_positive_weights(self):
        with self.assertRaises(ValueError):
            self._generator(year_weights={"2021": 0.0})


class ConsensusIgnoreTests(unittest.TestCase):
    """Objetivo consensus-ignore: sin loss donde solo marca una minoría."""

    def test_valid_channel_masks_minority_pixels(self):
        a = _lab((10, 20, 10, 40))
        b = _lab((10, 20, 10, 30))
        c = _lab((10, 20, 10, 20), (40, 50, 40, 50))
        t = merge_targets([a, b, c], "consensus-ignore")
        np.testing.assert_array_equal(t[:3], merge_targets([a, b, c], "consensus"))
        self.assertEqual(t[3][15, 15], 1.0)   # 3/3
        self.assertEqual(t[3][15, 25], 1.0)   # 2/3
        self.assertEqual(t[3][15, 35], 0.0)   # 1/3: disputado
        self.assertEqual(t[3][45, 45], 0.0)   # 1/3: disputado
        self.assertEqual(t[3][60, 60], 1.0)   # fondo claro

    def test_loss_ignores_masked_pixels(self):
        import torch
        from filaments.model import SegLoss
        target = torch.zeros(1, 4, 32, 32)
        target[:, 0, 8:16, 8:16] = 1.0
        target[:, 3] = 1.0
        target[:, 0, 20:24, 20:24] = 1 / 3
        target[:, 3, 20:24, 20:24] = 0.0
        logits = torch.randn(1, 3, 32, 32, requires_grad=True)
        loss = SegLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.grad[0, 0, 20:24, 20:24].abs().sum().item(), 0.0)
        self.assertGreater(logits.grad[0, 0, 8:16, 8:16].abs().sum().item(), 0.0)
        logits2 = logits.detach().clone()
        logits2[0, 0, 20:24, 20:24] += 5.0  # cambiar solo los píxeles ignorados no cambia la loss
        self.assertAlmostEqual(SegLoss()(logits2, target).item(), loss.item(), places=5)


if __name__ == "__main__":
    unittest.main()
