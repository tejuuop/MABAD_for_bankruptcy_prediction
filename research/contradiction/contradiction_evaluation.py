"""Run the standalone Track A feature-quality audit and validation comparison.

The script reads only the configured raw fields, fits numerical references on
the existing chronological training split, and writes new outputs under this
directory. It does not touch baseline artifacts or score the previously
examined test split.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .contradiction_score import TrackAScorer
    from .financial_topic_scores import load_topic_config, topic_definitions
except ImportError:  # Direct ``python contradiction_evaluation.py`` support.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research.contradiction.contradiction_score import TrackAScorer
    from research.contradiction.financial_topic_scores import load_topic_config, topic_definitions


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parents[1] / "data"
DEFAULT_PHASE4 = ROOT.parent / "phase 4" / "phase4_pre_registered" / "inputs" / "active_finbert_mabad_valtest_modality_distances.csv"
SEED = 20261011
N_SPLITS = 5


def configured_financial_columns(config: dict) -> list[str]:
    fields = []
    for topic in topic_definitions(config):
        for ratio in topic.ratios:
            fields.extend([ratio.numerator, ratio.denominator])
    return list(dict.fromkeys(fields))


def load_numerical_records(data_dir: Path, config: dict) -> pd.DataFrame:
    numerical_columns = ["uid", "label", *configured_financial_columns(config)]
    numerical = pd.read_csv(data_dir / "numerical.csv", usecols=numerical_columns)
    if numerical["uid"].duplicated().any():
        raise ValueError("Numerical identifiers must be unique")
    return numerical


def iter_record_chunks(data_dir: Path, numerical: pd.DataFrame, chunk_size: int = 128):
    """Stream long text while retaining exact numerical/text UID joins."""
    numeric_by_uid = numerical.set_index("uid", drop=False)
    seen: set[str] = set()
    for text in pd.read_csv(data_dir / "mda_auditor.csv", usecols=["uid", "item_7", "opinion_text"], chunksize=chunk_size):
        if text["uid"].duplicated().any() or seen.intersection(text["uid"].astype(str)):
            raise ValueError("Duplicate text identifier encountered")
        seen.update(text["uid"].astype(str))
        missing = set(text["uid"]) - set(numeric_by_uid.index)
        if missing:
            raise ValueError(f"Text identifiers missing from numerical data: {len(missing)}")
        numeric_part = numeric_by_uid.loc[text["uid"].tolist()].reset_index(drop=True)
        text = text.reset_index(drop=True)
        yield pd.concat([numeric_part.drop(columns=["item_7", "opinion_text"], errors="ignore"), text[["item_7", "opinion_text"]]], axis=1)
    if len(seen) != len(numerical) or seen != set(numerical["uid"].astype(str)):
        raise ValueError("Numerical/text UID sets do not match")


def existing_time_split(frame: pd.DataFrame) -> pd.Series:
    """Replicate utils.time_split using the UID date and positive boundaries."""
    dates = pd.to_datetime(frame["uid"].str.split("__", n=1).str[1], errors="coerce")
    if dates.isna().any():
        raise ValueError("Every UID must contain a parseable date suffix")
    positives = frame.loc[frame["label"].astype(bool)].assign(_date=dates)
    positives = positives.sort_values("_date").reset_index(drop=True)
    if len(positives) < 4:
        raise ValueError("Not enough positive rows for the existing split rule")
    train_end = int(0.65 * len(positives))
    val_end = int(0.75 * len(positives))
    train_max = positives.iloc[train_end]["_date"]
    val_max = positives.iloc[val_end]["_date"]
    split = pd.Series("test", index=frame.index, dtype="object")
    split.loc[dates <= train_max] = "train"
    split.loc[(dates > train_max) & (dates <= val_max)] = "val"
    return split


def _metric_row(name: str, y: np.ndarray, score: np.ndarray, evaluation: str, fold: int | None = None) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    k = min(20, len(y))
    row = {
        "model": name,
        "evaluation": evaluation,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "precision_at_20": float(y[order[:k]].sum() / k) if k else np.nan,
        "recall_at_20": float(y[order[:k]].sum() / max(1, int(y.sum()))) if k else np.nan,
    }
    if fold is not None:
        row["fold"] = fold
    return row


def _components_to_wide(features: pd.DataFrame, topic_names: list[str]) -> pd.DataFrame:
    rows = []
    for value in features["topic_components_json"]:
        parsed = json.loads(value)
        row = {}
        for topic in topic_names:
            payload = parsed[topic]
            for key in ["numerical_severity", "narrative_acknowledged_severity", "static_gap", "reverse_gap"]:
                row[f"{topic}__{key}"] = payload[key]
            row[f"{topic}__numerical_available"] = int(payload["numerical_available"])
            row[f"{topic}__narrative_available"] = int(payload["narrative_available"])
            row[f"{topic}__eligible"] = int(payload["eligible"])
        rows.append(row)
    return pd.DataFrame(rows, index=features.index)


def feature_quality(features: pd.DataFrame, records: pd.DataFrame, split: pd.Series, topic_names: list[str]) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "val", "test"]:
        mask = split.eq(split_name).to_numpy()
        part = features.loc[mask]
        row = {
            "split": split_name,
            "rows": int(len(part)),
            "positives": int(records.loc[mask, "label"].astype(int).sum()),
            "contradiction_available_rows": int(part["contradiction_score"].notna().sum()),
            "coverage_positive_rows": int((part["coverage"] > 0).sum()),
            "mean_coverage": float(part["coverage"].mean()),
            "median_coverage": float(part["coverage"].median()),
            "max_score": float(part["contradiction_score"].max()) if part["contradiction_score"].notna().any() else np.nan,
            "min_score": float(part["contradiction_score"].min()) if part["contradiction_score"].notna().any() else np.nan,
        }
        parsed = _components_to_wide(part, topic_names)
        for topic in topic_names:
            row[f"{topic}_numerical_available"] = int(parsed[f"{topic}__numerical_available"].sum())
            row[f"{topic}_narrative_available"] = int(parsed[f"{topic}__narrative_available"].sum())
            row[f"{topic}_eligible"] = int(parsed[f"{topic}__eligible"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _fusion_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
    ])


def controlled_validation_comparison(features: pd.DataFrame, phase4_path: Path, topic_names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Compare frozen held-out MABAD distances plus feature blocks on validation.

    The Phase 4 input contains MABAD distances produced before this experiment
    and applied to the held-out 2018 validation rows. It is therefore suitable
    for this validation-only comparison, but is not a new MABAD CV retraining.
    """
    if not phase4_path.exists():
        return pd.DataFrame(), {"status": "blocked", "reason": f"Missing artifact: {phase4_path}"}
    baseline = pd.read_csv(phase4_path)
    required = {"uid", "company_id", "split", "label", "d_max"}
    missing = required - set(baseline.columns)
    if missing:
        return pd.DataFrame(), {"status": "blocked", "reason": f"Phase 4 artifact missing columns: {sorted(missing)}"}
    val = baseline[baseline["split"].eq("val")].copy()
    join = features[["uid", "contradiction_score", "coverage", "eligible_topic_count", "topic_components_json"]].copy()
    val = val.merge(join, on="uid", how="left", validate="one_to_one")
    if len(val) != len(baseline[baseline["split"].eq("val")]) or val["contradiction_score"].isna().all():
        return pd.DataFrame(), {"status": "blocked", "reason": "Feature join failed or produced no contradiction scores"}
    wide = _components_to_wide(val, topic_names)
    val = pd.concat([val.reset_index(drop=True), wide.reset_index(drop=True)], axis=1)
    y = val["label"].astype(int).to_numpy()
    groups = val["company_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    base_direct = _metric_row("mabad_d_max_frozen", y, val["d_max"].to_numpy(float), "frozen_validation_baseline")
    rows = [base_direct]
    oof = {}
    blocks = {
        "mabad_plus_contradiction_coverage": ["contradiction_score", "coverage", "eligible_topic_count"],
        "mabad_plus_severity_without_gap": [
            c for topic in topic_names for c in [
                f"{topic}__numerical_severity", f"{topic}__narrative_acknowledged_severity",
                f"{topic}__numerical_available", f"{topic}__narrative_available",
            ]
        ],
        "mabad_plus_coverage_missingness_only": ["coverage", "eligible_topic_count"],
    }
    for model_name, extra_columns in blocks.items():
        pred = np.full(len(val), np.nan)
        fold_rows = []
        for fold, (train_idx, held_idx) in enumerate(splitter.split(val, y, groups), start=1):
            train = val.iloc[train_idx]
            held = val.iloc[held_idx]
            columns = ["d_max", *extra_columns]
            model = _fusion_pipeline()
            model.fit(train[columns], train["label"].astype(int))
            pred[held_idx] = model.predict_proba(held[columns])[:, 1]
            fold_rows.append(_metric_row(model_name, held["label"], pred[held_idx], "grouped_validation_fold", fold))
        if np.isnan(pred).any():
            raise RuntimeError(f"Missing OOF predictions for {model_name}")
        oof[model_name] = pred
        rows.append(_metric_row(model_name, y, pred, "pooled_grouped_validation_oof"))
        for fold_row in fold_rows:
            rows.append(fold_row)
    result = pd.DataFrame(rows)
    metadata = {
        "status": "completed_validation_only",
        "baseline_source": str(phase4_path),
        "validation_rows": int(len(val)),
        "validation_positives": int(y.sum()),
        "cv_splitter": "StratifiedGroupKFold(company_id)",
        "test_metrics": "not calculated; existing test split was previously examined",
        "mabad_prediction_status": "frozen held-out validation distances from existing artifact; no MABAD retraining performed",
        "claim_extractor_status": "rules_based_unvalidated_proxy",
    }
    return result, metadata


def write_outputs(data_dir: Path, output_dir: Path, phase4_path: Path) -> None:
    config = load_topic_config()
    topic_names = [t.name for t in topic_definitions(config)]
    records = load_numerical_records(data_dir, config)
    split = existing_time_split(records)
    train = records.loc[split.eq("train")].copy()
    scorer = TrackAScorer(config)
    scorer.fit(train)
    feature_parts = []
    evidence_parts = []
    numerical_parts = []
    for record_chunk in iter_record_chunks(data_dir, records):
        features_chunk, evidence_chunk, numerical_chunk = scorer.transform(record_chunk)
        feature_parts.append(features_chunk)
        if not evidence_chunk.empty:
            evidence_parts.append(evidence_chunk)
        numerical_parts.append(numerical_chunk)
    features = pd.concat(feature_parts, ignore_index=True)
    evidence = pd.concat(evidence_parts, ignore_index=True) if evidence_parts else pd.DataFrame()
    numerical = pd.concat(numerical_parts, ignore_index=True)

    if len(features) != len(records) or set(features["uid"]) != set(records["uid"]):
        raise AssertionError("Feature output does not preserve input identifiers")
    if not features["uid"].is_unique:
        raise AssertionError("Feature output contains duplicate identifiers")
    if not features["label"].equals(records["label"].astype(int).reset_index(drop=True)):
        raise AssertionError("Feature output labels do not align with input rows")
    if features["contradiction_score"].notna().any():
        score = features["contradiction_score"].dropna().to_numpy(float)
        if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
            raise AssertionError("Contradiction scores must be finite and bounded")
    if ((features["coverage"] < 0) | (features["coverage"] > 1)).any():
        raise AssertionError("Coverage must be bounded")

    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "contradiction_features.csv", index=False)
    evidence.to_csv(output_dir / "contradiction_evidence.csv", index=False)
    numerical_out = pd.concat([features[["uid", "company_id", "prediction_cutoff"]].reset_index(drop=True), numerical.reset_index(drop=True)], axis=1)
    numerical_out.to_csv(output_dir / "financial_topic_components.csv", index=False)
    quality = feature_quality(features, records, split, topic_names)
    quality.to_csv(output_dir / "feature_quality_summary.csv", index=False)

    comparison, comparison_meta = controlled_validation_comparison(features, phase4_path, topic_names)
    if not comparison.empty:
        comparison.to_csv(output_dir / "controlled_validation_comparison.csv", index=False)
    (output_dir / "controlled_validation_metadata.json").write_text(json.dumps(comparison_meta, indent=2), encoding="utf-8")

    schema = {
        "score_version": config["score_version"],
        "narrative_status": config["narrative_status"],
        "temporal_scoring": config["temporal_scoring"],
        "feature_outputs": {
            "contradiction_score": "weighted mean of eligible static under-acknowledgement gaps; null if no eligible topic",
            "coverage": "eligible topic weight divided by configured topic weight; not confidence",
            "topic_components_json": "topic-level B, A, S, T=null, R, evidence IDs, availability, and limitations",
            "missing_topics": "configured topics without both numerical severity and applicable current claim",
        },
        "evidence_outputs": {
            "contradiction_evidence.csv": "management candidate sentences and separate auditor evidence",
            "financial_topic_components.csv": "raw ratio values, training-only severity percentiles, and availability",
        },
        "ratio_definitions": [
            {
                "topic": topic.name,
                "name": ratio.name,
                "numerator": ratio.numerator,
                "denominator": ratio.denominator,
                "adverse_direction": ratio.adverse_direction,
                "denominator_rule": ratio.denominator_rule,
            }
            for topic in topic_definitions(config) for ratio in topic.ratios
        ],
        "split_definition": "Existing repository positive-boundary chronological split using UID date; UID date is not verified publication date.",
        "fit_rows": int(len(train)),
        "all_rows": int(len(records)),
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    run_meta = {
        "status": "static_track_a_mvp",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "rows": int(len(records)),
        "split_counts": split.value_counts().to_dict(),
        "positive_counts": records.assign(_split=split).groupby("_split")["label"].sum().to_dict(),
        "evidence_rows": int(len(evidence)),
        "management_evidence_rows": int((evidence.get("source_field", pd.Series(dtype=str)) == "item_7").sum()),
        "auditor_evidence_rows": int((evidence.get("source_field", pd.Series(dtype=str)) == "opinion_text").sum()),
        "fit_uids_not_used_for_transform_refit": True,
        "test_metrics": "not calculated",
        "limitations": [
            "No verified filing/publication date; temporal contradiction scoring is disabled.",
            "Narrative acknowledgement values are transparent rules-based proxies without human validation.",
            "Configured XBRL concepts are sparse and heterogeneous across companies.",
            "Existing later test data was previously examined and is not reported as an untouched holdout.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
    report = [
        "# Track A MVP — Static contradiction feature audit",
        "",
        "Status: implementation prototype. Numerical severity uses training-only empirical percentiles. Narrative acknowledgement is a rules-based, unvalidated proxy.",
        "",
        "## Data and temporal boundary",
        "",
        f"- Rows: {len(records):,}; companies: {records['uid'].str.split('__').str[0].nunique():,}.",
        f"- Split counts: {split.value_counts().to_dict()}.",
        "- UID dates are used only for the repository's existing chronological split and display. No verified filing/publication date is available.",
        "- Temporal contradiction score T is disabled and emitted as null.",
        "",
        "## Feature quality",
        "",
        "```\n" + quality.to_string(index=False) + "\n```",
        "",
        "## Controlled MABAD comparison",
        "",
    ]
    if comparison.empty:
        report.append(f"Blocked: {comparison_meta.get('reason', 'no comparison output')}.")
    else:
        pooled = comparison[comparison["evaluation"].str.contains("pooled", na=False)]
        report.append("The comparison uses frozen held-out MABAD distances from the existing Phase 4 input and grouped validation only. No test metrics were calculated.")
        report.append("")
        report.append("```\n" + pooled.to_string(index=False) + "\n```")
    report += [
        "",
        "## Interpretation limits",
        "",
        "- A high score means the configured numerical severity exceeds the extracted acknowledgement proxy; it is not a probability and does not imply deception.",
        "- Missing claims remain unavailable rather than being treated as reassurance.",
        "- Auditor evidence is retained separately and is not included in management acknowledgement A.",
        "- These outputs do not establish novelty, validity, or predictive improvement.",
    ]
    (output_dir / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "run_metadata": run_meta, "comparison": comparison_meta}, indent=2, default=str))


def write_report_from_existing(output_dir: Path) -> None:
    """Finish a run from already-written outputs without rescanning full MD&A."""
    quality_path = output_dir / "feature_quality_summary.csv"
    comparison_path = output_dir / "controlled_validation_comparison.csv"
    metadata_path = output_dir / "run_metadata.json"
    if not quality_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Existing feature outputs are incomplete")
    quality = pd.read_csv(quality_path)
    run_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = [
        "# Track A MVP — Static contradiction feature audit",
        "",
        "Status: implementation prototype. Numerical severity uses training-only empirical percentiles. Narrative acknowledgement is a rules-based, unvalidated proxy.",
        "",
        "## Data and temporal boundary",
        "",
        f"- Rows: {run_meta['rows']:,}.",
        f"- Split counts: {run_meta['split_counts']}.",
        "- UID dates are used only for the repository's existing chronological split and display. No verified filing/publication date is available.",
        "- Temporal contradiction score T is disabled and emitted as null.",
        "",
        "## Feature quality",
        "",
        "```\n" + quality.to_string(index=False) + "\n```",
        "",
        "## Controlled MABAD comparison",
        "",
    ]
    if not comparison_path.exists():
        report.append("Blocked: controlled comparison output is unavailable.")
    else:
        comparison = pd.read_csv(comparison_path)
        pooled = comparison[comparison["evaluation"].str.contains("pooled", na=False)]
        report.append("Frozen held-out MABAD distances were compared on grouped validation only. No test metrics were calculated.")
        report.append("")
        report.append("```\n" + pooled.to_string(index=False) + "\n```")
    report += [
        "",
        "## Interpretation limits",
        "",
        "- A high score means configured numerical severity exceeds the extracted acknowledgement proxy; it is not a probability and does not imply deception.",
        "- Missing claims remain unavailable rather than being treated as reassurance.",
        "- Auditor evidence is retained separately and is not included in management acknowledgement A.",
        "- These outputs do not establish novelty, validity, or predictive improvement.",
    ]
    (output_dir / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "status": "report_written_from_existing_outputs"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--phase4-input", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--report-existing", action="store_true", help="Write the report from completed output tables without rescanning text")
    args = parser.parse_args()
    if args.report_existing:
        write_report_from_existing(args.output_dir)
    else:
        write_outputs(args.data_dir, args.output_dir, args.phase4_input)


if __name__ == "__main__":
    main()
