#!/usr/bin/env python3
"""Phase 4 pre-specified calibration robustness audit.

Primary analysis: training-fold-only transformations and pooled OOF metrics on the
2018 validation split. No label-dependent fitting or weight selection. The later
test split is used ONLY for split/company/time-overlap audit, not method selection.

Run from this directory:
  python phase4_calibration_audit.py
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "active_finbert_mabad_valtest_modality_distances.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MODS = ["d_financial", "d_mda", "d_auditor"]
SEED = 20261012
N_SPLITS = 5
B = 2000

def percentile_matrix(reference, frame):
    """Train-reference empirical CDF; values are in (0,1), higher means more anomalous."""
    out = []
    for c in MODS:
        ref = np.sort(reference[c].astype(float).to_numpy())
        vals = frame[c].astype(float).to_numpy()
        # Mid-rank style empirical percentile; reference is fit only on training fold.
        out.append(np.searchsorted(ref, vals, side="right") / (len(ref) + 1.0))
    return np.column_stack(out)

def robust_z_matrix(reference, frame):
    """Robustly standardize by training-fold median/IQR; preserve high=more anomalous."""
    med = reference[MODS].median().to_numpy(float)
    q1 = reference[MODS].quantile(.25).to_numpy(float)
    q3 = reference[MODS].quantile(.75).to_numpy(float)
    scale = q3 - q1
    # Avoid exploding scores for near-constant modalities.
    sd = reference[MODS].std(ddof=0).to_numpy(float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, sd)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    z = (frame[MODS].to_numpy(float) - med) / scale
    # Distances are nonnegative and larger is more anomalous; clip low tail only.
    return np.maximum(z, 0.0)

def score_set(train, held):
    raw = held[MODS].to_numpy(float)
    pct = percentile_matrix(train, held)
    rz = robust_z_matrix(train, held)
    # Pre-specified, label-independent candidates; no weights are learned from labels.
    return {
        "raw_max_original": raw.max(axis=1),
        "raw_mean": raw.mean(axis=1),
        "raw_power_mean_p4": np.mean(np.maximum(raw, 0.0)**4, axis=1)**0.25,
        "percentile_max": pct.max(axis=1),
        "percentile_mean": pct.mean(axis=1),
        "percentile_power_mean_p2": np.mean(pct**2, axis=1)**0.5,
        "percentile_power_mean_p4": np.mean(pct**4, axis=1)**0.25,
        "percentile_top2_mean": np.sort(pct, axis=1)[:, -2:].mean(axis=1),
        "robust_z_max_clipped": rz.max(axis=1),
        "robust_z_mean_clipped": rz.mean(axis=1),
    }

def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

def metrics(name, y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    npos = int(y.sum())
    row = {"method": name, "n": int(len(y)), "positives": npos,
           "prevalence": float(y.mean()),
           "average_precision": float(average_precision_score(y, s)) if npos else float("nan"),
           "roc_auc": safe_auc(y, s)}
    for k in [20, 50, 100]:
        kk = min(k, len(y))
        row[f"precision_at_{k}"] = float(y[order[:kk]].sum() / kk)
        row[f"recall_at_{k}"] = float(y[order[:kk]].sum() / max(1, npos))
    return row

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input missing: {INPUT}")
    df = pd.read_csv(INPUT)
    required = {"split", "label", *MODS}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=["split", "label", *MODS]).copy()
    df["label"] = df["label"].astype(int)
    df["split_norm"] = df["split"].astype(str).str.lower()
    val = df[df["split_norm"].isin(["val", "validation"])].copy().reset_index(drop=True)
    test = df[df["split_norm"].eq("test")].copy().reset_index(drop=True)
    if val.empty or test.empty: raise ValueError("Expected validation and test splits.")
    y = val.label.to_numpy(int)
    if val.label.nunique() != 2: raise ValueError("Validation must contain both classes.")
    if "company_id" not in val:
        raise ValueError("company_id is required for group-aware CV.")

    # Explicit split/time/company overlap audit; test labels/metrics are not used for selection.
    shared = set(val.company_id.astype(str)) & set(test.company_id.astype(str))
    audit = {
        "input_file": INPUT.name, "seed": SEED, "n_splits": N_SPLITS,
        "validation_rows": int(len(val)), "validation_positives": int(y.sum()),
        "validation_year_counts": {str(k): int(v) for k,v in val.year.value_counts(dropna=False).items()} if "year" in val else {},
        "validation_unique_companies": int(val.company_id.nunique()),
        "validation_repeated_company_rows": int(val.company_id.duplicated(keep=False).sum()),
        "test_rows": int(len(test)), "test_positives": int(test.label.sum()),
        "test_year_counts": {str(k): int(v) for k,v in test.year.value_counts(dropna=False).items()} if "year" in test else {},
        "company_ids_shared_between_validation_and_test": int(len(shared)),
        "shared_company_overlap_note": "Rows for the same company may occur in different years/splits. This is an audit finding, not evidence of feature leakage by itself.",
        "primary_metric": "pooled out-of-fold average precision",
        "candidate_weights_or_transformations_fit_using_labels": False,
        "test_metrics_used_for_method_selection": False,
        "caution": "All candidate comparisons are exploratory; only 19 validation positives are present, and candidates were not independently confirmed."
    }
    if "date" in val:
        audit["validation_date_min"] = str(val.date.min())
        audit["validation_date_max"] = str(val.date.max())
    if "year" in val and "year" in test:
        audit["validation_test_year_overlap"] = sorted(set(val.year.dropna().astype(str)) & set(test.year.dropna().astype(str)))

    # Diagnostics: distances, scale ratios, within-label summaries are descriptive only.
    diag = []
    for c in MODS:
        for name, part in [("validation_all", val), ("validation_negative", val[val.label == 0]),
                           ("validation_positive", val[val.label == 1])]:
            x = part[c].astype(float)
            diag.append({"modality": c, "subset": name, "n": len(x), "mean": x.mean(),
                         "std": x.std(), "median": x.median(), "iqr": x.quantile(.75)-x.quantile(.25),
                         "unique_values": x.nunique()})
    pd.DataFrame(diag).to_csv(OUT/"distance_scale_diagnostics.csv", index=False)
    val[["company_id","label"] + (["year"] if "year" in val else [])].groupby("company_id").agg(
        rows=("label","size"), positives=("label","sum")).reset_index().query("rows > 1").to_csv(
        OUT/"repeated_company_validation_rows.csv", index=False)

    # Company-stratified folds, if available; otherwise fail rather than silently fall back.
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(val, y, groups=val.company_id.astype(str).to_numpy()))
    oof = {}
    fold_rows = []
    fold_ids = np.full(len(val), -1, int)
    for fold, (tr_idx, ho_idx) in enumerate(folds, 1):
        tr = val.iloc[tr_idx].reset_index(drop=True)
        ho = val.iloc[ho_idx].reset_index(drop=True)
        cand = score_set(tr, ho)
        if not oof:
            oof = {m: np.full(len(val), np.nan) for m in cand}
        for m, s in cand.items():
            oof[m][ho_idx] = s
            row = metrics(m, ho.label.to_numpy(), s); row["fold"] = fold
            fold_rows.append(row)
        fold_ids[ho_idx] = fold
    if (fold_ids < 0).any() or any(np.isnan(s).any() for s in oof.values()):
        raise RuntimeError("Missing out-of-fold predictions.")

    pred = val.copy()
    pred["oof_fold"] = fold_ids
    for m, s in oof.items(): pred["oof__" + m] = s
    pred.to_csv(OUT/"validation_oof_predictions.csv", index=False)
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT/"validation_fold_metrics.csv", index=False)
    fold_summary = folds_df.groupby("method").agg(
        fold_mean_ap=("average_precision","mean"), fold_sd_ap=("average_precision","std"),
        fold_mean_auc=("roc_auc","mean"), fold_sd_auc=("roc_auc","std")).reset_index()
    pooled = pd.DataFrame([metrics(m, y, s) for m,s in oof.items()])
    summary = pooled.merge(fold_summary, on="method").sort_values("average_precision", ascending=False)
    summary["evaluation"] = "pooled_oof_validation_2018"
    summary.to_csv(OUT/"validation_oof_summary.csv", index=False)

    # Cluster bootstrap by company, preserving all rows from each sampled company.
    # Since the 2018 validation split has almost one row/company, this mostly acts
    # like paired stratified row bootstrap but remains correct if repeated groups exist.
    rng = np.random.default_rng(SEED)
    companies = val.company_id.astype(str).to_numpy()
    unique_groups = np.unique(companies)
    group_rows = {g: np.flatnonzero(companies == g) for g in unique_groups}
    # Stratify company clusters by whether they contain at least one positive.
    pos_groups = np.array([g for g in unique_groups if y[group_rows[g]].sum() > 0])
    neg_groups = np.array([g for g in unique_groups if y[group_rows[g]].sum() == 0])
    boot_rows = []
    base = "raw_max_original"
    for m in oof:
        if m == base: continue
        diffs = []
        for _ in range(B):
            sampled_pos = rng.choice(pos_groups, size=len(pos_groups), replace=True)
            sampled_neg = rng.choice(neg_groups, size=len(neg_groups), replace=True)
            sampled = np.concatenate([sampled_pos, sampled_neg])
            idx = np.concatenate([group_rows[g] for g in sampled])
            if len(np.unique(y[idx])) < 2: continue
            diffs.append(average_precision_score(y[idx], oof[m][idx]) -
                         average_precision_score(y[idx], oof[base][idx]))
        boot_rows.append({
            "method": m, "baseline": base,
            "observed_oof_ap_difference": float(average_precision_score(y,oof[m])-average_precision_score(y,oof[base])),
            "bootstrap_replicates_requested": B, "bootstrap_replicates_used": len(diffs),
            "bootstrap_ci_2_5": float(np.quantile(diffs,.025)) if diffs else np.nan,
            "bootstrap_ci_97_5": float(np.quantile(diffs,.975)) if diffs else np.nan,
            "note": "Paired cluster bootstrap by company, stratified by positive-bearing company; exploratory uncertainty, not independent confirmation."
        })
    pd.DataFrame(boot_rows).to_csv(OUT/"validation_cluster_bootstrap.csv", index=False)

    # Test is used for temporal/split audit only. No test model metrics are calculated.
    if "year" in test:
        test.groupby("year").agg(rows=("label","size"), positives=("label","sum"),
                                 companies=("company_id","nunique")).reset_index().to_csv(
            OUT/"test_year_counts_audit_only.csv", index=False)

    audit["cv_splitter"] = "StratifiedGroupKFold(company_id)"
    audit["candidate_count"] = len(oof)
    audit["top_method_by_oof_ap"] = str(summary.iloc[0].method)
    audit["top_oof_ap"] = float(summary.iloc[0].average_precision)
    audit["baseline_oof_ap"] = float(summary.loc[summary.method == "raw_max_original","average_precision"].iloc[0])
    audit["top_minus_baseline_oof_ap"] = audit["top_oof_ap"] - audit["baseline_oof_ap"]
    (OUT/"phase4_run_metadata.json").write_text(json.dumps(audit, indent=2))
    print("\nPHASE 4 COMPLETE")
    print(summary[["method","average_precision","roc_auc","precision_at_20","recall_at_20","fold_mean_ap","fold_sd_ap"]].to_string(index=False))
    print("\nValidation rows/positives:", len(val), int(y.sum()))
    print("Company IDs shared across validation/test:", len(shared))
    print("IMPORTANT: no test performance was calculated or used for selection.")
    print("Outputs:")
    for p in sorted(OUT.iterdir()): print(" -", p.name)

if __name__ == "__main__":
    main()
