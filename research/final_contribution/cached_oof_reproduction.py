#!/usr/bin/env python3
"""Reproduce the fixed Phase 4 max-versus-power-mean comparison from cached OOF rows.

This script is intentionally read-only with respect to prior research artifacts. It
loads only the saved validation OOF prediction table, recomputes two fixed scores
from its three modality distances, and writes new audit outputs in this workspace.
It never loads or evaluates the project test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def power_mean(values: np.ndarray, power: float) -> np.ndarray:
    """Return the non-negative generalized mean across modalities."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional row-by-modality matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("power mean requires finite, non-negative distances")
    if power <= 0:
        raise ValueError("power must be positive for this implementation")
    return np.mean(values**power, axis=1) ** (1.0 / power)


def score_metrics(name: str, labels: np.ndarray, scores: np.ndarray, budgets: Iterable[int]) -> dict[str, float | int | str]:
    """Calculate ranking metrics with deterministic tie ordering."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("labels and finite scores must have equal length")
    positives = int(labels.sum())
    if positives == 0 or len(np.unique(labels)) != 2:
        raise ValueError("metrics require at least one positive and one negative row")
    order = np.argsort(-scores, kind="mergesort")
    result: dict[str, float | int | str] = {
        "method": name,
        "n": int(len(labels)),
        "positives": positives,
        "prevalence": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    for budget in budgets:
        k = min(int(budget), len(labels))
        captured = int(labels[order[:k]].sum())
        result[f"precision_at_{budget}"] = captured / k
        result[f"recall_at_{budget}"] = captured / positives
    return result


def paired_company_cluster_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    proposal: np.ndarray,
    companies: np.ndarray,
    replicates: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Paired bootstrap of AP(proposal)-AP(baseline), preserving company rows."""
    labels = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline, dtype=float)
    proposal = np.asarray(proposal, dtype=float)
    companies = np.asarray(companies, dtype=str)
    unique_companies = np.unique(companies)
    rows_by_company = {company: np.flatnonzero(companies == company) for company in unique_companies}
    positive_companies = np.array(
        [company for company in unique_companies if labels[rows_by_company[company]].sum() > 0], dtype=str
    )
    negative_companies = np.array(
        [company for company in unique_companies if labels[rows_by_company[company]].sum() == 0], dtype=str
    )
    if len(positive_companies) == 0 or len(negative_companies) == 0:
        raise ValueError("bootstrap requires positive-bearing and negative-only company clusters")
    if rng is None:
        if seed is None:
            raise ValueError("either seed or rng is required")
        rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_positive = rng.choice(positive_companies, size=len(positive_companies), replace=True)
        sampled_negative = rng.choice(negative_companies, size=len(negative_companies), replace=True)
        sampled_rows = np.concatenate([rows_by_company[company] for company in np.concatenate([sampled_positive, sampled_negative])])
        differences[replicate] = (
            average_precision_score(labels[sampled_rows], proposal[sampled_rows])
            - average_precision_score(labels[sampled_rows], baseline[sampled_rows])
        )
    return differences


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict, Path]:
    config_path = path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config, config_path.parent


def main(config_path: Path) -> None:
    config, config_dir = load_config(config_path)
    source = (config_dir / config["source_phase4_oof_predictions"]).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Cached validation OOF predictions are missing: {source}")
    if str(config.get("test_set_usage", "")).lower() != "prohibited":
        raise ValueError("This reproduction must prohibit test-set usage")

    modalities = list(config["modalities"])
    baseline_name = str(config["baseline_name"])
    proposal_name = str(config["proposal_name"])
    saved_baseline = f"oof__{baseline_name}"
    saved_proposal = f"oof__{proposal_name}"
    frame = pd.read_csv(source)
    required = {"uid", "company_id", "label", "oof_fold", *modalities, saved_baseline, saved_proposal}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cached OOF file is missing columns: {missing}")
    if frame.uid.duplicated().any():
        raise ValueError("Cached OOF file contains duplicate UIDs")
    if "split" in frame and not frame["split"].astype(str).str.lower().isin(["val", "validation"]).all():
        raise ValueError("Cached OOF source must contain validation rows only")
    labels = frame.label.to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must be binary with both classes present")
    if (frame.oof_fold.to_numpy(dtype=int) < 1).any():
        raise ValueError("OOF fold identifiers must be positive")

    distances = frame[modalities].to_numpy(dtype=float)
    baseline = np.max(distances, axis=1)
    proposal = power_mean(distances, float(config["power"]))
    if not np.allclose(baseline, frame[saved_baseline].to_numpy(dtype=float), rtol=1e-12, atol=1e-12):
        raise ValueError("Recomputed maximum differs from the saved Phase 4 baseline score")
    if not np.allclose(proposal, frame[saved_proposal].to_numpy(dtype=float), rtol=1e-12, atol=1e-12):
        raise ValueError("Recomputed power mean differs from the saved Phase 4 proposal score")
    if (proposal > baseline + 1e-12).any():
        raise ValueError("Power mean cannot exceed the maximum for non-negative distances")

    budgets = [int(value) for value in config["review_budgets"]]
    summary = pd.DataFrame(
        [
            score_metrics(baseline_name, labels, baseline, budgets),
            score_metrics(proposal_name, labels, proposal, budgets),
        ]
    )
    folds: list[dict[str, float | int | str]] = []
    for fold in sorted(frame.oof_fold.unique()):
        indices = frame.oof_fold.to_numpy(dtype=int) == int(fold)
        for name, scores in [(baseline_name, baseline), (proposal_name, proposal)]:
            row = score_metrics(name, labels[indices], scores[indices], budgets)
            row["fold"] = int(fold)
            folds.append(row)
    fold_metrics = pd.DataFrame(folds)

    # Phase 4 used one RNG and generated raw-mean draws before p=4 draws.
    # Preserve that sequence so this cached audit can verify its saved interval.
    shared_rng = np.random.default_rng(int(config["bootstrap_seed"]))
    _ = paired_company_cluster_bootstrap(
        labels=labels,
        baseline=baseline,
        proposal=distances.mean(axis=1),
        companies=frame.company_id.astype(str).to_numpy(),
        replicates=int(config["bootstrap_replicates"]),
        rng=shared_rng,
    )
    bootstrap = paired_company_cluster_bootstrap(
        labels=labels,
        baseline=baseline,
        proposal=proposal,
        companies=frame.company_id.astype(str).to_numpy(),
        replicates=int(config["bootstrap_replicates"]),
        rng=shared_rng,
    )
    observed_difference = float(
        summary.loc[summary.method == proposal_name, "average_precision"].iloc[0]
        - summary.loc[summary.method == baseline_name, "average_precision"].iloc[0]
    )
    reference_bootstrap = (config_dir / config["source_phase4_bootstrap"]).resolve()
    if not reference_bootstrap.exists():
        raise FileNotFoundError(f"Saved Phase 4 bootstrap output is missing: {reference_bootstrap}")
    reference = pd.read_csv(reference_bootstrap)
    reference_row = reference.loc[reference.method.eq(proposal_name)]
    if len(reference_row) != 1:
        raise ValueError("Saved Phase 4 bootstrap output has no unique p=4 row")
    reference_row = reference_row.iloc[0]
    reproduced_ci_low = float(np.quantile(bootstrap, 0.025))
    reproduced_ci_high = float(np.quantile(bootstrap, 0.975))
    bootstrap_summary = pd.DataFrame(
        [
            {
                "method": proposal_name,
                "baseline": baseline_name,
                "observed_ap_difference": observed_difference,
                "replicates": int(len(bootstrap)),
                "bootstrap_seed": int(config["bootstrap_seed"]),
                "bootstrap_mean_difference": float(bootstrap.mean()),
                "bootstrap_ci_2_5": reproduced_ci_low,
                "bootstrap_ci_97_5": reproduced_ci_high,
                "probability_difference_gt_zero": float((bootstrap > 0).mean()),
                "saved_phase4_observed_ap_difference": float(reference_row["observed_oof_ap_difference"]),
                "saved_phase4_ci_2_5": float(reference_row["bootstrap_ci_2_5"]),
                "saved_phase4_ci_97_5": float(reference_row["bootstrap_ci_97_5"]),
                "matches_saved_phase4_bootstrap": bool(
                    np.isclose(observed_difference, float(reference_row["observed_oof_ap_difference"]), rtol=1e-12, atol=1e-12)
                    and np.isclose(reproduced_ci_low, float(reference_row["bootstrap_ci_2_5"]), rtol=1e-12, atol=1e-12)
                    and np.isclose(reproduced_ci_high, float(reference_row["bootstrap_ci_97_5"]), rtol=1e-12, atol=1e-12)
                ),
                "procedure": "paired company-cluster bootstrap; clusters stratified by positive-bearing status",
                "interpretation": "exploratory validation uncertainty; not independent confirmation",
            }
        ]
    )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    used = frame[["uid", "company_id", "label", "oof_fold", *modalities]].copy()
    if "date" in frame:
        used.insert(2, "date", frame.date)
    used[baseline_name] = baseline
    used[proposal_name] = proposal
    used.to_csv(output_dir / "cached_oof_predictions_used.csv", index=False)
    summary.to_csv(output_dir / "validation_metrics_reproduced.csv", index=False)
    fold_metrics.to_csv(output_dir / "validation_fold_metrics_reproduced.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "paired_company_cluster_bootstrap_reproduced.csv", index=False)
    pd.DataFrame({"ap_difference": bootstrap}).to_csv(output_dir / "paired_company_cluster_bootstrap_draws.csv", index=False)

    metadata = {
        "status": "completed_cached_prediction_reproduction",
        "source_file": str(source),
        "source_sha256": file_sha256(source),
        "saved_phase4_bootstrap_source": str(reference_bootstrap),
        "saved_phase4_bootstrap_sha256": file_sha256(reference_bootstrap),
        "source_rows": int(len(frame)),
        "source_positive_rows": int(labels.sum()),
        "source_unique_companies": int(frame.company_id.astype(str).nunique()),
        "source_duplicate_uids": int(frame.uid.duplicated().sum()),
        "folds": sorted(int(value) for value in frame.oof_fold.unique()),
        "baseline": baseline_name,
        "proposal": proposal_name,
        "power": float(config["power"]),
        "score_match_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        "scores_recomputed_from_frozen_modality_distances": True,
        "bootstrap_sequence_matches_original_phase4_method_order_through_p4": True,
        "test_labels_or_test_metrics_loaded": False,
        "test_set_usage": "prohibited",
        "selection_warning": "p=4 was selected within prior exploratory validation candidate work; this audit does not establish independent selection or generalization.",
    }
    (output_dir / "reproduction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(bootstrap_summary.to_string(index=False))
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    arguments = parser.parse_args()
    main(arguments.config)
