"""Invariantes de muestreo y evaluación, sin GPU ni datos del concurso."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from filaments.sampling import select_observations
from rescore_ablation import bootstrap_delta, observation_counts, score, subset
from rescore_train import kept_by_threshold


class SamplingTests(unittest.TestCase):
    def test_proportions_and_reproducibility(self):
        stems = [f"{year}{i:010d}{station}h" for year in (2011, 2022)
                 for station in "BC" for i in range(20)]
        selected = select_observations(stems, 24, 7)
        self.assertEqual(selected, select_observations(stems[::-1], 24, 7))
        self.assertEqual(len(set(selected)), 24)
        for year in (2011, 2022):
            for station in "BC":
                self.assertEqual(sum(s.startswith(str(year)) and s.endswith(station + "h")
                                     for s in selected), 6)
        self.assertNotEqual(selected, select_observations(stems, 24, 8))

    def test_limits_and_small_strata(self):
        stems = [f"{2010+i}0101000000Bh" for i in range(12)]
        self.assertEqual(len(select_observations(stems, 3)), 3)
        self.assertEqual(select_observations(stems, 0), sorted(stems))
        self.assertEqual(select_observations(stems, 100), sorted(stems))
        self.assertEqual(select_observations(stems, 3, strategy="chronological"), sorted(stems)[:3])
        with self.assertRaises(ValueError):
            select_observations(stems, -1)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"stem": ["a", "a", "b"], "id": [1, 2, 1],
                                "match": [np.array([.7, 0]), np.array([0., 0]), np.array([.8])]})
        self.gts = {"a": [1, 1], "b": [1], "empty": [2]}

    def test_multi_annotator_and_empty_observation(self):
        c = observation_counts(self.df, self.gts, np.array([True, False, True]))
        # a: TP .7 + un FP y un FN; b: TP .8; empty: dos FN.
        np.testing.assert_allclose(np.sum(list(c.values()), axis=0), [1.5, 2, 1, 3])
        self.assertAlmostEqual(score(list(c.values())), .375)
        a = np.stack(list(c.values()))
        self.assertEqual(bootstrap_delta(a, a, 1), [0., 0.])

    def test_minimum_and_subset_preserve_observations(self):
        scores = np.array([.2, .1, .05])
        np.testing.assert_array_equal(kept_by_threshold(self.df, scores, .9, 0), [False]*3)
        np.testing.assert_array_equal(kept_by_threshold(self.df, scores, .9, 1), [True, False, True])
        df, gts = subset(self.df, self.gts, ["b", "empty"])
        self.assertEqual(df.stem.tolist(), ["b"])
        self.assertEqual(set(gts), {"b", "empty"})


if __name__ == "__main__":
    unittest.main()
