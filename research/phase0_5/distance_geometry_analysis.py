#!/usr/bin/env python3
"""
Controlled MABAD distance-geometry ablation.

Research question
-----------------
Does the relative geometry of the three modality anomaly distances
(financial, MD&A, auditor) add information beyond the original MABAD
d_max score?

This is deliberately a SMALL, pre-specified ablation. It is not a
feature-search exercise.

Protocol
--------
1. Load the existing Phase 0 validation/test modality-distance output.
2. Fit every model on validation only.
3. Freeze the fitted model/scaler.
4. Evaluate on the untouched test set.
5. Use 5-fold stratified CV inside validation only as a stability check.
6. Use a class-stratified bootstrap on the frozen test predictions for
   paired AP differences.

Models
------
G0  d_max
G1  d_max + max_minus_min_range
G2  d_max + disagreement_std
G3  d_max + range + disagreement
G4  raw three modality distances
G5  d_max + dominant-modality indicators

The most important comparison is G0 vs G1.
If G1 survives, the simple range feature is much easier to defend
than a broad collection of engineered features.

Run from repository root:
    python research/phase0_5/distance_geometry_analysis.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Phase 0.5's diagnostic file is expected to contain the modality
# distances and derived geometry features.
INPUT_CANDIDATES = [
    REPO_ROOT / "research/phase0/outputs/active_finbert_mabad_valtest_modality_distances.csv",
    REPO_ROOT / "research/phase0/outputs/active_finbert_mabad_valtest_diagnostics.csv",
]

OUTPUT_DIR = REPO_ROOT / "research/phase0_5/distance_geometry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_LIST = [10, 20, 50, 100]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find the Phase 0 modality-distance/diagnostic file.\n"
        "Expected one of:\n"
        + "\n".join(str(p) for p in INPUT_CANDIDATES)
    )


def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(f"  - {c}" for c in df.columns)
        )


def ranking_metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "ap": float(average_precision_score(y, score)),
        "auc": float(roc_auc_score(y, score)),
    }

    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    total_pos = int(y.sum())

    for k in K_LIST:
        kk = min(k, len(ys))
        hits = int(ys[:kk].sum())
        out[f"precision_at_{k}"] = hits / kk
        out[f"recall_at_{k}"] = hits / total_pos if total_pos else np.nan

    return out


def make_model(features):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_on_validation_predict(model_features, val, eval_df):
    model = make_model(model_features)
    model.fit(val[model_features], val["label"])
    score = model.predict_proba(eval_df[model_features])[:, 1]
    return model, score


def coefficient_table(model, features):
    scaler = model.named_steps["scale"]
    lr = model.named_steps["lr"]

    return [
        {
            "feature": feature,
            "standardized_coefficient": float(coef),
            "odds_ratio_per_1sd": float(np.exp(coef)),
            "validation_mean": float(mu),
            "validation_sd": float(sd),
        }
        for feature, coef, mu, sd in zip(
            features, lr.coef_[0], scaler.mean_, scaler.scale_
        )
    ]


def stratified_bootstrap_ap_difference(
    y,
    score_base,
    score_enhanced,
    n_boot=5000,
    seed=RANDOM_STATE,
):
    """
    Paired class-stratified bootstrap.

    Positives and negatives are resampled separately, preserving the
    observed class counts in every replicate. This avoids the invalid
    practice of discarding ordinary bootstrap samples with zero positives.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    score_base = np.asarray(score_base, dtype=float)
    score_enhanced = np.asarray(score_enhanced, dtype=float)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    diffs = np.empty(n_boot)

    for i in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([p, n])

        ap_base = average_precision_score(y[idx], score_base[idx])
        ap_enh = average_precision_score(y[idx], score_enhanced[idx])
        diffs[i] = ap_enh - ap_base

    return {
        "mean_difference": float(diffs.mean()),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "probability_enhanced_beats_base": float(np.mean(diffs > 0)),
        "n_boot": n_boot,
    }


def validation_cv(df, model_specs):
    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), start=1):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        for name, features in model_specs.items():
            model = make_model(features)
            model.fit(tr[features], tr["label"])
            score = model.predict_proba(va[features])[:, 1]

            rows.append(
                {
                    "fold": fold,
                    "model": name,
                    **ranking_metrics(va["label"], score),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    input_file = find_input()
    df = pd.read_csv(input_file)

    # The Phase 0 implementation uses these names. If a slightly
    # different artifact is supplied, fail loudly rather than guessing.
    required = [
        "split",
        "label",
        "d_financial",
        "d_mda",
        "d_auditor",
        "d_max",
        "distance_max_minus_min",
        "dominant_modality",
    ]
    require_columns(df, required)

    # The Phase 0 modality-distance CSV does not store disagreement_std.
    # Reconstruct it directly from the three saved modality distances.
    # This is the population standard deviation across the three modality
    # anomaly distances for each filing, matching the Phase 0 definition
    # of cross-modal dispersion.
    df["disagreement_std"] = df[
        ["d_financial", "d_mda", "d_auditor"]
    ].std(axis=1, ddof=0)

    df = df.dropna(
        subset=required + ["disagreement_std"]
    ).copy()
    df["label"] = df["label"].astype(int)

    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    if val["label"].sum() < 2 or test["label"].sum() < 2:
        raise ValueError("Validation/test does not contain enough positives.")

    # -----------------------------------------------------------------
    # Dominant-modality indicators
    # -----------------------------------------------------------------
    for modality in ["financial", "mda", "auditor"]:
        df[f"dom_{modality}"] = (
            df["dominant_modality"].astype(str).str.lower()
            == modality
        ).astype(int)

    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    # -----------------------------------------------------------------
    # Pre-specified model families
    # -----------------------------------------------------------------
    model_specs = {
        "G0_dmax": [
            "d_max",
        ],
        "G1_dmax_plus_range": [
            "d_max",
            "distance_max_minus_min",
        ],
        "G2_dmax_plus_disagreement": [
            "d_max",
            "disagreement_std",
        ],
        "G3_dmax_plus_range_plus_disagreement": [
            "d_max",
            "distance_max_minus_min",
            "disagreement_std",
        ],
        "G4_raw_three_distances": [
            "d_financial",
            "d_mda",
            "d_auditor",
        ],
        "G5_dmax_plus_dominant_modality": [
            "d_max",
            "dom_financial",
            "dom_mda",
            "dom_auditor",
        ],
    }

    results = {
        "input_file": str(input_file),
        "protocol": {
            "fit_split": "validation",
            "evaluation_split": "test",
            "test_used_for_fitting": False,
            "test_used_for_scaling": False,
            "test_used_for_model_selection": False,
            "random_state": RANDOM_STATE,
            "bootstrap": "paired class-stratified bootstrap",
        },
        "dataset": {
            "validation_n": int(len(val)),
            "validation_positives": int(val["label"].sum()),
            "test_n": int(len(test)),
            "test_positives": int(test["label"].sum()),
        },
        "models": {},
    }

    test_scores = pd.DataFrame(
        {
            "label": test["label"].to_numpy(),
        },
        index=test.index,
    )

    coefficient_rows = []

    # -----------------------------------------------------------------
    # Frozen validation -> test evaluation
    # -----------------------------------------------------------------
    fitted = {}

    for name, features in model_specs.items():
        model, test_score = fit_on_validation_predict(
            features, val, test
        )

        # Validation score from the SAME final validation-fitted model.
        val_score = model.predict_proba(val[features])[:, 1]

        fitted[name] = model
        test_scores[name] = test_score

        results["models"][name] = {
            "features": features,
            "validation_metrics": ranking_metrics(val["label"], val_score),
            "test_metrics": ranking_metrics(test["label"], test_score),
            "coefficients": coefficient_table(model, features),
        }

        for row in coefficient_table(model, features):
            coefficient_rows.append(
                {
                    "model": name,
                    **row,
                }
            )

    # -----------------------------------------------------------------
    # Paired bootstrap for the most important comparisons
    # -----------------------------------------------------------------
    comparisons = [
        ("G0_dmax", "G1_dmax_plus_range"),
        ("G0_dmax", "G2_dmax_plus_disagreement"),
        ("G0_dmax", "G3_dmax_plus_range_plus_disagreement"),
        ("G0_dmax", "G4_raw_three_distances"),
        ("G0_dmax", "G5_dmax_plus_dominant_modality"),
    ]

    results["test_incremental_comparisons"] = {}

    y_test = test["label"].to_numpy()

    for base, enhanced in comparisons:
        results["test_incremental_comparisons"][
            f"{base}_vs_{enhanced}"
        ] = stratified_bootstrap_ap_difference(
            y_test,
            test_scores[base].to_numpy(),
            test_scores[enhanced].to_numpy(),
        )

    # -----------------------------------------------------------------
    # Validation CV stability
    # -----------------------------------------------------------------
    cv = validation_cv(val, model_specs)
    cv.to_csv(OUTPUT_DIR / "validation_5fold_results.csv", index=False)

    cv_summary = (
        cv.groupby("model")[
            [
                "ap",
                "auc",
                "precision_at_10",
                "precision_at_20",
                "precision_at_50",
                "precision_at_100",
                "recall_at_10",
                "recall_at_20",
                "recall_at_50",
                "recall_at_100",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv_summary.to_csv(
        OUTPUT_DIR / "validation_5fold_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Save raw outputs
    # -----------------------------------------------------------------
    test_scores.to_csv(
        OUTPUT_DIR / "test_frozen_predictions.csv",
        index=False,
    )

    pd.DataFrame(coefficient_rows).to_csv(
        OUTPUT_DIR / "model_coefficients.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "distance_geometry_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------------------
    # Human-readable report
    # -----------------------------------------------------------------
    lines = [
        "# MABAD Distance-Geometry Ablation",
        "",
        "## Research question",
        "",
        "Does the relative geometry of financial, MD&A, and auditor anomaly "
        "distances add information beyond the original MABAD `d_max` score?",
        "",
        "## Protocol",
        "",
        "- Every model is fitted on validation only.",
        "- Feature scaling is fitted on validation only.",
        "- Test labels are never used for fitting, scaling, or selection.",
        "- Five-fold CV is performed only inside validation as a stability check.",
        "- Final test results are frozen holdout evaluations.",
        "",
        "## Pre-specified models",
        "",
        "| Model | Features | Purpose |",
        "|---|---|---|",
        "| G0 | d_max | Original MABAD baseline |",
        "| G1 | d_max + max-min range | **Primary geometry test** |",
        "| G2 | d_max + disagreement | Disagreement replication |",
        "| G3 | d_max + range + disagreement | Combined geometry |",
        "| G4 | three raw modality distances | Direct modality-distance test |",
        "| G5 | d_max + dominant modality | Tests whether modality identity matters |",
        "",
        "## Frozen test results",
        "",
        "| Model | AP | AUC | P@20 | P@50 | P@100 | R@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in model_specs:
        m = results["models"][name]["test_metrics"]
        lines.append(
            f"| {name} | {m['ap']:.4f} | {m['auc']:.4f} | "
            f"{m['precision_at_20']:.4f} | "
            f"{m['precision_at_50']:.4f} | "
            f"{m['precision_at_100']:.4f} | "
            f"{m['recall_at_100']:.4f} |"
        )

    lines += [
        "",
        "## Paired class-stratified bootstrap comparisons",
        "",
        "Positive AP difference means the enhanced model beats d_max.",
        "",
    ]

    for key, comp in results["test_incremental_comparisons"].items():
        lines.append(
            f"### {key}\n"
            f"- Mean AP difference: {comp['mean_difference']:.4f}\n"
            f"- 95% CI: [{comp['ci95_low']:.4f}, "
            f"{comp['ci95_high']:.4f}]\n"
            f"- Bootstrap P(enhanced > base): "
            f"{comp['probability_enhanced_beats_base']:.4f}\n"
        )

    lines += [
        "## Interpretation rules",
        "",
        "### Strong evidence",
        "G1 improves over G0 on validation and frozen test, with a positive "
        "bootstrap interval and consistent Top-K improvement.",
        "",
        "### Moderate evidence",
        "G1 improves on test but the interval touches/crosses zero, or "
        "validation/test evidence is inconsistent. Treat as a lead only.",
        "",
        "### Redundant/negative",
        "G1 does not consistently beat G0. Do not claim that distance geometry "
        "is an incremental contribution.",
        "",
        "### Important caution",
        "G4/G5 are secondary ablations. They must not be used to cherry-pick "
        "a winning feature combination after seeing test results.",
        "",
    ]

    with open(
        OUTPUT_DIR / "distance_geometry_report.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines))

    print("\n".join(lines[:40]))
    print("\nOutputs written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
