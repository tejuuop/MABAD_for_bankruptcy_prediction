#!/usr/bin/env python3
"""
Phase 0.5 follow-up:
Controlled incremental-value test for cross-modal disagreement.

Question:
Does disagreement_std add predictive information beyond the existing
MABAD d_max score?

Protocol:
1. Read Phase 0 diagnostics only.
2. Fit all model parameters using VALIDATION only.
3. Use TEST exactly once as a frozen holdout evaluation.
4. Compare:
   M0: d_max
   M1: disagreement_std
   M2: d_max + disagreement_std
   M3: d_max + disagreement_std + distance_max_minus_min (ablation)
5. Also run stratified 5-fold CV on validation as a stability check.
6. No test labels are used for fitting, feature scaling, thresholds, or
   model selection.

Run from the repository root:
    python research/phase0_5/incremental_value_analysis.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "research/phase0/outputs/active_finbert_mabad_valtest_diagnostics.csv"
OUTPUT_DIR = REPO_ROOT / "research/phase0_5/incremental_value"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_LIST = [10, 20, 50, 100]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns:\n"
            + ", ".join(df.columns)
        )


def ranking_metrics(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)

    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "ap": np.nan,
        "auc": np.nan,
    }

    if len(y) and y.sum() > 0 and y.sum() < len(y):
        out["ap"] = float(average_precision_score(y, score))
        out["auc"] = float(roc_auc_score(y, score))

    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]

    total_pos = int(y.sum())
    for k in K_LIST:
        kk = min(k, len(y_sorted))
        found = int(y_sorted[:kk].sum())
        out[f"precision_at_{k}"] = float(found / kk) if kk else np.nan
        out[f"recall_at_{k}"] = float(found / total_pos) if total_pos else np.nan

    return out


def make_pipeline(n_features):
    # No class weighting: this is a ranking/incremental-information test,
    # and weighting would introduce another modeling choice.
    return Pipeline(
        steps=[
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


def fit_predict(train_df, eval_df, features):
    model = make_pipeline(len(features))
    model.fit(train_df[features], train_df["label"])
    score = model.predict_proba(eval_df[features])[:, 1]
    return model, score


def coefficient_table(model, features):
    scaler = model.named_steps["scale"]
    lr = model.named_steps["lr"]

    rows = []
    for feature, coef, mean, scale in zip(
        features,
        lr.coef_[0],
        scaler.mean_,
        scaler.scale_,
    ):
        rows.append(
            {
                "feature": feature,
                "standardized_coefficient": float(coef),
                "odds_ratio_per_1sd": float(np.exp(coef)),
                "validation_mean": float(mean),
                "validation_sd": float(scale),
            }
        )
    return rows


def stratified_bootstrap_metric_difference(
    y,
    score_a,
    score_b,
    n_boot=5000,
    seed=42,
):
    """
    Class-stratified bootstrap for AP and AP difference.

    Positives and negatives are resampled separately, so every replicate
    contains the observed number of positives and negatives. This avoids
    the invalid practice of dropping ordinary bootstrap replicates that
    happen to contain zero positives.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    if len(pos) == 0 or len(neg) == 0:
        return {}

    diffs = np.empty(n_boot, dtype=float)
    ap_a = np.empty(n_boot, dtype=float)
    ap_b = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx_pos = rng.choice(pos, size=len(pos), replace=True)
        idx_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([idx_pos, idx_neg])

        aa = average_precision_score(y[idx], score_a[idx])
        bb = average_precision_score(y[idx], score_b[idx])

        ap_a[i] = aa
        ap_b[i] = bb
        diffs[i] = bb - aa

    return {
        "ap_a_mean": float(ap_a.mean()),
        "ap_b_mean": float(ap_b.mean()),
        "ap_difference_b_minus_a_mean": float(diffs.mean()),
        "ap_difference_b_minus_a_ci95_low": float(np.percentile(diffs, 2.5)),
        "ap_difference_b_minus_a_ci95_high": float(np.percentile(diffs, 97.5)),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_type": "class-stratified bootstrap",
    }


def validation_cv_comparison(df, feature_sets):
    """
    Five-fold stratified validation CV.

    This is only a stability check. It does not replace the final
    validation-fit -> frozen-test protocol.
    """
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

        for model_name, features in feature_sets.items():
            model, score = fit_predict(tr, va, features)
            mets = ranking_metrics(va["label"], score)
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    **mets,
                }
            )

    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}\n\n"
            "Run this script from your MABAD repository, and make sure "
            "the Phase 0 output exists."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "split",
        "label",
        "d_max",
        "disagreement_std",
        "distance_max_minus_min",
    ]
    require_columns(df, required)

    df = df.dropna(
        subset=["label", "d_max", "disagreement_std", "distance_max_minus_min"]
    ).copy()
    df["label"] = df["label"].astype(int)

    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    if val["label"].sum() < 2:
        raise ValueError("Validation set has too few positives.")
    if test["label"].sum() < 2:
        raise ValueError("Test set has too few positives.")

    feature_sets = {
        "M0_dmax": ["d_max"],
        "M1_disagreement": ["disagreement_std"],
        "M2_dmax_plus_disagreement": ["d_max", "disagreement_std"],
        "M3_dmax_plus_disagreement_plus_range": [
            "d_max",
            "disagreement_std",
            "distance_max_minus_min",
        ],
    }

    results = {
        "protocol": {
            "question": (
                "Does cross-modal disagreement provide incremental "
                "predictive information beyond MABAD d_max?"
            ),
            "fit_split": "validation",
            "evaluation_split": "test",
            "test_used_for_fitting": False,
            "test_used_for_feature_scaling": False,
            "test_used_for_model_selection": False,
            "random_state": RANDOM_STATE,
        },
        "dataset": {
            "validation_n": int(len(val)),
            "validation_positives": int(val["label"].sum()),
            "test_n": int(len(test)),
            "test_positives": int(test["label"].sum()),
        },
        "models": {},
    }

    all_coef_rows = []
    test_score_table = test[["label"]].copy()

    fitted_models = {}

    # ---------------------------------------------------------------
    # Final validation-fit -> test evaluation
    # ---------------------------------------------------------------
    for model_name, features in feature_sets.items():
        model, val_score = fit_predict(val, val, features)
        _, test_score = fit_predict(val, test, features)

        # Reuse the same fitted model for validation predictions above.
        # fit_predict(val, val, ...) created the final validation-fit model.
        model = model
        fitted_models[model_name] = model

        val_metrics = ranking_metrics(val["label"], val_score)
        test_metrics = ranking_metrics(test["label"], test_score)

        results["models"][model_name] = {
            "features": features,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "coefficients": coefficient_table(model, features),
        }

        for row in coefficient_table(model, features):
            row["model"] = model_name
            all_coef_rows.append(row)

        test_score_table[f"{model_name}_score"] = test_score

    # ---------------------------------------------------------------
    # Incremental AP differences on frozen test predictions
    # ---------------------------------------------------------------
    comparisons = [
        ("M0_dmax", "M2_dmax_plus_disagreement"),
        ("M0_dmax", "M3_dmax_plus_disagreement_plus_range"),
        ("M1_disagreement", "M2_dmax_plus_disagreement"),
    ]

    for base_name, enhanced_name in comparisons:
        base_score = test_score_table[f"{base_name}_score"].to_numpy()
        enhanced_score = test_score_table[f"{enhanced_name}_score"].to_numpy()

        boot = stratified_bootstrap_metric_difference(
            test["label"].to_numpy(),
            base_score,
            enhanced_score,
            n_boot=5000,
            seed=RANDOM_STATE,
        )

        results.setdefault("test_incremental_comparisons", {})[
            f"{base_name}_vs_{enhanced_name}"
        ] = {
            "base_model": base_name,
            "enhanced_model": enhanced_name,
            **boot,
        }

    # ---------------------------------------------------------------
    # Validation 5-fold stability check
    # ---------------------------------------------------------------
    cv = validation_cv_comparison(val, feature_sets)
    cv.to_csv(OUTPUT_DIR / "validation_5fold_results.csv", index=False)

    cv_summary = (
        cv.groupby("model")[
            ["ap", "auc", "precision_at_10", "precision_at_20",
             "precision_at_50", "precision_at_100",
             "recall_at_10", "recall_at_20", "recall_at_50", "recall_at_100"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv_summary.to_csv(OUTPUT_DIR / "validation_5fold_summary.csv", index=False)

    # ---------------------------------------------------------------
    # Save detailed outputs
    # ---------------------------------------------------------------
    test_score_table.to_csv(
        OUTPUT_DIR / "test_frozen_predictions.csv",
        index=False,
    )

    pd.DataFrame(all_coef_rows).to_csv(
        OUTPUT_DIR / "model_coefficients.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "incremental_value_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------------------------------------------------------------
    # Human-readable report
    # ---------------------------------------------------------------
    lines = [
        "# Controlled Incremental-Value Analysis",
        "",
        "## Research question",
        "",
        "Does cross-modal disagreement provide predictive information beyond "
        "the existing MABAD `d_max` anomaly score?",
        "",
        "## Protocol",
        "",
        "- Model parameters and feature scaling were fitted on validation only.",
        "- Test labels were not used for fitting, scaling, or model selection.",
        "- Final test predictions are therefore frozen holdout predictions.",
        "- A 5-fold stratified CV on validation is included only as a stability check.",
        "",
        "## Models",
        "",
        "- **M0:** `d_max`",
        "- **M1:** `disagreement_std`",
        "- **M2:** `d_max + disagreement_std`",
        "- **M3:** `d_max + disagreement_std + distance_max_minus_min`",
        "",
        "## Final test results",
        "",
        "| Model | AP | AUC | P@10 | P@20 | P@50 | P@100 | R@10 | R@20 | R@50 | R@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for model_name in feature_sets:
        m = results["models"][model_name]["test_metrics"]
        lines.append(
            f"| {model_name} | "
            f"{m['ap']:.4f} | {m['auc']:.4f} | "
            f"{m['precision_at_10']:.4f} | {m['precision_at_20']:.4f} | "
            f"{m['precision_at_50']:.4f} | {m['precision_at_100']:.4f} | "
            f"{m['recall_at_10']:.4f} | {m['recall_at_20']:.4f} | "
            f"{m['recall_at_50']:.4f} | {m['recall_at_100']:.4f} |"
        )

    lines += [
        "",
        "## Incremental test comparisons",
        "",
    ]

    for name, comp in results["test_incremental_comparisons"].items():
        lines.append(
            f"### {name}\n"
            f"- Mean AP difference (enhanced - base): "
            f"{comp['ap_difference_b_minus_a_mean']:.4f}\n"
            f"- 95% class-stratified bootstrap CI: "
            f"[{comp['ap_difference_b_minus_a_ci95_low']:.4f}, "
            f"{comp['ap_difference_b_minus_a_ci95_high']:.4f}]\n"
        )

    lines += [
        "## Interpretation rule",
        "",
        "The key comparison is **M0 vs M2**.",
        "",
        "- If M2 consistently improves AP/Top-K performance over M0 and the "
        "bootstrap interval for AP improvement is clearly above zero, this "
        "supports incremental information from disagreement.",
        "- If M2 is nearly identical to M0, disagreement is probably redundant "
        "with the existing anomaly score.",
        "- If M2 improves validation but not frozen test, treat the result as "
        "unstable rather than claiming a contribution.",
        "- M3 is an ablation showing whether adding the simple max-min distance "
        "range explains the apparent disagreement effect.",
        "",
        "## Important limitation",
        "",
        "This experiment establishes predictive incremental value (or lack of it). "
        "It does not by itself establish that disagreement is a causal or "
        "mechanistically independent 'reliability' signal.",
        "",
    ]

    with open(OUTPUT_DIR / "incremental_value_report.md", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines[:45]))
    print("\nOutputs written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
