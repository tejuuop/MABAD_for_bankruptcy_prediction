import unittest

import numpy as np

from research.final_contribution.cached_oof_reproduction import power_mean, score_metrics


class CachedOofReproductionTests(unittest.TestCase):
    def test_power_mean_is_finite_and_between_mean_and_max(self):
        distances = np.array([[0.1, 0.2, 0.3], [0.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
        score = power_mean(distances, 4)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.all(score >= distances.mean(axis=1)))
        self.assertTrue(np.all(score <= distances.max(axis=1)))

    def test_power_mean_rejects_invalid_distances(self):
        for invalid in [np.array([[1.0, -1.0, 0.5]]), np.array([[1.0, np.nan, 0.5]])]:
            with self.assertRaises(ValueError):
                power_mean(invalid, 4)

    def test_metrics_are_reproducible_and_include_review_budget_recall(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        first = score_metrics("example", labels, scores, [2])
        second = score_metrics("example", labels, scores, [2])
        self.assertEqual(first, second)
        self.assertEqual(first["recall_at_2"], 0.5)
        self.assertGreaterEqual(first["average_precision"], 0.0)
        self.assertLessEqual(first["average_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
