import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filaments.validation import compare_reports, public_sample_diagnostic, pq_totals


def report():
    return {"name": "example", "seed": 7, "folds": 2, "train_stems": ["train"],
            "validation_stems": ["a", "b"], "variants": {"match_year": {
                "folds": [{"fold": 0, "development_stems": ["b"], "evaluation_stems": ["a"]},
                          {"fold": 1, "development_stems": ["a"], "evaluation_stems": ["b"]}],
                "policies": {"1": {"per_observation": {"a": [.6, 1, 0, 0], "b": [1.2, 2, 0, 0]}}}}}}


class DiagnosticsTests(unittest.TestCase):
    def test_aggregate_not_mean(self):
        self.assertAlmostEqual(float(pq_totals(np.array([[.6, 1, 0, 0], [0, 0, 0, 3]]).sum(0))), .24)

    def test_same_quality_has_zero_sampling_spread(self):
        d = public_sample_diagnostic(report(), size=90, repetitions=100, reference_pq=.39)
        np.testing.assert_allclose(d["bootstrap_95"], [.6, .6])
        self.assertEqual(d["empirical_fraction_at_or_below_reference"], 0.)

    def test_pair_identity_and_reordering(self):
        a, b = report(), report()
        b["validation_stems"].reverse()
        d = compare_reports(a, b, repetitions=100)
        self.assertEqual(d["paired_bootstrap_95"], [0., 0.])
        self.assertFalse(d["positive_lower_bound"])

    def test_incompatible_reports_rejected(self):
        reference = report()
        for change in ("seed", "fold", "gt", "leak"):
            other = copy.deepcopy(reference)
            if change == "seed":
                other["seed"] = 99
            elif change == "fold":
                other["variants"]["match_year"]["folds"].reverse()
            elif change == "gt":
                other["variants"]["match_year"]["policies"]["1"]["per_observation"]["a"][3] = 1
            else:
                other["train_stems"].append("a")
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_reports(other, reference, repetitions=100)


if __name__ == "__main__":
    unittest.main()
