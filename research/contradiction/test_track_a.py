"""Automated behavioral and leakage-boundary tests for the static MVP."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from .contradiction_score import TrackAScorer
from .narrative_claims import NarrativeClaimExtractor


def _row(uid: str, cash: float, assets: float, current_assets: float, current_liabilities: float,
         liabilities: float, debt: float, interest: float, revenue: float, net_income: float,
         operating_income: float, cfo: float, text: str, auditor: str = "") -> dict:
    return {
        "uid": uid,
        "label": 0,
        "CashAndCashEquivalentsAtCarryingValue": cash,
        "Assets": assets,
        "AssetsCurrent": current_assets,
        "LiabilitiesCurrent": current_liabilities,
        "Liabilities": liabilities,
        "LongTermDebt": debt,
        "InterestExpense": interest,
        "Revenues": revenue,
        "NetIncomeLoss": net_income,
        "OperatingIncomeLoss": operating_income,
        "NetCashProvidedByUsedInOperatingActivities": cfo,
        "item_7": text,
        "opinion_text": auditor,
    }


def _train_and_examples() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame([
        _row(f"C{i}__201{i}-12-31", 500 + i * 20, 1000, 500, 250, 300, 150, 20, 1000, 100, 120, 150, "Operations were stable.")
        for i in range(5)
    ])
    examples = pd.DataFrame([
        _row("A__2020-12-31", 5, 1000, 20, 200, 990, 900, 300, 100, -300, -250, -200, "Our liquidity is strong and adequate.", "The auditor discussed liquidity."),
        _row("B__2020-12-31", 5, 1000, 20, 200, 990, 900, 300, 100, -300, -250, -200, "We face significant liquidity pressure and debt challenges."),
        _row("C__2020-12-31", 5, 1000, 20, 200, 990, 900, 300, 100, -300, -250, -200, "The company acquired a liquidity management business."),
        _row("D__2020-12-31", 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, "Our liquidity is strong."),
        _row("E__2020-12-31", 5, 1000, 20, 200, 990, 900, 300, 100, -300, -250, -200, "Liquidity is not a concern."),
    ])
    return train, examples


class TrackATests(unittest.TestCase):
    def test_high_numerical_severity_with_acknowledgement_has_lower_gap(self):
        train, examples = _train_and_examples()
        scorer = TrackAScorer().fit(train)
        features, evidence, _ = scorer.transform(examples.iloc[[0, 1]])
        self.assertTrue(features.loc[0, "contradiction_score"] >= 0)
        self.assertLess(features.loc[1, "contradiction_score"], features.loc[0, "contradiction_score"])
        self.assertGreater(features.loc[1, "coverage"], 0)

    def test_missing_narrative_is_unavailable_not_optimistic(self):
        train, examples = _train_and_examples()
        examples.loc[2, "item_7"] = "The company operates a portfolio of offices."
        features, _, _ = TrackAScorer().fit(train).transform(examples.iloc[[2]])
        self.assertTrue(pd.isna(features.iloc[0]["contradiction_score"]))
        self.assertEqual(features.iloc[0]["coverage"], 0.0)

    def test_missing_numerical_fields_and_zero_denominators_are_unavailable(self):
        train, examples = _train_and_examples()
        features, _, numeric = TrackAScorer().fit(train).transform(examples.iloc[[3]])
        self.assertTrue(pd.isna(features.iloc[0]["contradiction_score"]))
        self.assertEqual(features.iloc[0]["coverage"], 0.0)
        self.assertEqual(int(numeric.iloc[0]["liquidity__numerical_available"]), 0)

    def test_negation_and_misleading_keyword_are_auditable(self):
        train, examples = _train_and_examples()
        extractor = NarrativeClaimExtractor()
        claims = extractor.extract(examples.iloc[[2, 4]])
        self.assertFalse((claims["filing_uid"] == "C__2020-12-31").any())
        negated = claims[claims["filing_uid"] == "E__2020-12-31"]
        self.assertTrue(len(negated) >= 1)
        self.assertTrue(bool(negated.iloc[0]["negated"]))
        self.assertIn("sentence", claims.columns)

    def test_missing_topics_and_zero_coverage(self):
        train, examples = _train_and_examples()
        examples.loc[0, "item_7"] = "Our liquidity is strong."
        features, _, _ = TrackAScorer().fit(train).transform(examples.iloc[[0]])
        self.assertEqual(features.iloc[0]["coverage"], 1 / 3)
        self.assertIn("leverage", features.iloc[0]["missing_topics"])

    def test_scores_are_bounded_finite_and_reproducible(self):
        train, examples = _train_and_examples()
        scorer = TrackAScorer().fit(train)
        first, _, _ = scorer.transform(examples)
        second, _, _ = scorer.transform(examples)
        np.testing.assert_allclose(first["coverage"], second["coverage"])
        valid = first["contradiction_score"].dropna().to_numpy(float)
        self.assertTrue(np.isfinite(valid).all())
        self.assertTrue(((valid >= 0) & (valid <= 1)).all())

    def test_training_only_transform_does_not_change_when_future_row_is_added(self):
        train, examples = _train_and_examples()
        scorer = TrackAScorer().fit(train)
        old, _, _ = scorer.transform(examples.iloc[[0]])
        future = examples.iloc[[0]].copy()
        future.loc[future.index[0], "uid"] = "A__2099-12-31"
        future.loc[future.index[0], "CashAndCashEquivalentsAtCarryingValue"] = 0.0
        combined, _, _ = scorer.transform(pd.concat([examples.iloc[[0]], future], ignore_index=True))
        self.assertEqual(old.loc[0, "contradiction_score"], combined.loc[0, "contradiction_score"])
        self.assertTrue(all(not name.startswith("temporal") for name in combined.columns))

    def test_auditor_evidence_is_separate(self):
        train, examples = _train_and_examples()
        features, evidence, _ = TrackAScorer().fit(train).transform(examples.iloc[[0]])
        auditor = evidence[evidence["source_field"] == "opinion_text"]
        self.assertGreaterEqual(len(auditor), 1)
        self.assertTrue((auditor["evidence_role"] == "separate_auditor_evidence_not_in_A").all())


if __name__ == "__main__":
    unittest.main()
