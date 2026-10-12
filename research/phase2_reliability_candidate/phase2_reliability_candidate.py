#!/usr/bin/env python3
"""
Phase 2 candidate: scale-aware reliability-weighted percentile aggregation.

Important:
- This is an exploratory post-hoc analysis of saved Phase 0.5 distances.
- It does NOT retrain MABAD and does NOT require imputer.pkl.
- Weights/reference distributions are fitted on validation/fold-training data only.
- The provided test split has been examined in earlier project work; its metrics
  are exploratory, not an untouched confirmation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "active_finbert_mabad_valtest_modality_distances.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODS = ["d_financial", "d_mda", "d_auditor"]

def percentile_matrix(reference, frame):
    cols = []
    for c in MODS:
        ref = np.sort(reference[c].to_numpy(float))
        vals = frame[c].to_numpy(float)
        cols.append(np.searchsorted(ref, vals, side="right") / (len(ref) + 1.0))
    return np.column_stack(cols)

def robust_weights(reference, floor=0.0):
    # IQR is measured on the fold-training/validation distribution only.
    # Floor defaults to zero; this is a sensitivity-analysis candidate, not
    # a learned label-supervised weight.
    q = np.array([reference[c].quantile(.75) - reference[c].quantile(.25) for c in MODS], float)
    q = np.maximum(q, floor)
    if not np.isfinite(q).all() or q.sum() <= 0:
        return np.ones(len(MODS)) / len(MODS)
    return q / q.sum()

def weighted_score(ranks, weights):
    return ranks @ weights

def metric_row(name, split, y, score):
    order = np.argsort(-score, kind="mergesort")
    out = {
        "method": name, "split": split, "n": int(len(y)),
        "positives": int(np.sum(y)), "prevalence": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
    }
    for k in (20, 50, 100):
        kk = min(k, len(y))
        out[f"precision_at_{k}"] = float(np.sum(y[order[:kk]]) / kk)
        out[f"recall_at_{k}"] = float(np.sum(y[order[:kk]]) / max(1, int(np.sum(y))))
    return out

def main():
    df = pd.read_csv(INPUT)
    required = {"split", "label", *MODS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=["split", "label", *MODS]).copy()
    df["label"] = df["label"].astype(int)
    val = df[df["split"].astype(str).str.lower().isin(["val", "validation"])].reset_index(drop=True)
    test = df[df["split"].astype(str).str.lower().eq("test")].reset_index(drop=True)
    if val.empty or test.empty:
        raise ValueError("Expected split labels 'val'/'validation' and 'test'.")
    yv, yt = val.label.to_numpy(), test.label.to_numpy()
    rv, rt = percentile_matrix(val, val), percentile_matrix(val, test)

    equal_w = np.ones(3) / 3
    iqr_w = robust_weights(val)
    # A conservative pre-specified ablation: damp the low-variation financial
    # modality but keep the two text modalities equal. This is a hypothesis test.
    damped_w = np.array([0.1, 1.0, 1.0]); damped_w /= damped_w.sum()

    rows = []
    for split, part, ranks, y in [("validation", val, rv, yv), ("test_exploratory", test, rt, yt)]:
        raw = part[MODS].to_numpy(float)
        rows.append(metric_row("original_raw_max", split, y, raw.max(axis=1)))
        rows.append(metric_row("raw_mean", split, y, raw.mean(axis=1)))
        rows.append(metric_row("rank_equal_mean", split, y, weighted_score(ranks, equal_w)))
        rows.append(metric_row("rank_iqr_weighted", split, y, weighted_score(ranks, iqr_w)))
        rows.append(metric_row("rank_financial_damped_0.1_1_1", split, y, weighted_score(ranks, damped_w)))
    pd.DataFrame(rows).to_csv(OUT / "holdout_metrics_exploratory.csv", index=False)

    # Stratified 5-fold CV on validation. Each fold computes percentile refs and
    # IQR weights from fold-training rows only, then scores the held-out fold.
    # With only 19 positives total, fold estimates will be noisy.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    cv_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(val, yv), start=1):
        tr, va = val.iloc[tr_idx], val.iloc[va_idx]
        ranks = percentile_matrix(tr, va)
        raw = va[MODS].to_numpy(float)
        weights = robust_weights(tr)
        candidates = {
            "original_raw_max": raw.max(axis=1),
            "raw_mean": raw.mean(axis=1),
            "rank_equal_mean": weighted_score(ranks, equal_w),
            "rank_iqr_weighted": weighted_score(ranks, weights),
            "rank_financial_damped_0.1_1_1": weighted_score(ranks, damped_w),
        }
        for name, score in candidates.items():
            row = metric_row(name, f"validation_fold_{fold}", va.label.to_numpy(), score)
            row["fold"] = fold
            row["weights_financial"] = float(weights[0])
            row["weights_mda"] = float(weights[1])
            row["weights_auditor"] = float(weights[2])
            cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT / "validation_cv_fold_metrics.csv", index=False)
    summary = cv_df.groupby("method").agg(
        mean_ap=("average_precision", "mean"),
        sd_ap=("average_precision", "std"),
        mean_auc=("roc_auc", "mean"),
        sd_auc=("roc_auc", "std"),
    ).reset_index().sort_values("mean_ap", ascending=False)
    summary.to_csv(OUT / "validation_cv_summary.csv", index=False)

    diagnostic = {
        "rows": int(len(df)),
        "validation_rows": int(len(val)),
        "validation_positives": int(yv.sum()),
        "test_rows": int(len(test)),
        "test_positives": int(yt.sum()),
        "validation_iqr": {
            c: float(val[c].quantile(.75) - val[c].quantile(.25)) for c in MODS
        },
        "validation_iqr_weights": {c: float(w) for c, w in zip(MODS, iqr_w)},
        "fixed_damped_weights": {c: float(w) for c, w in zip(MODS, damped_w)},
        "warning": "Test metrics are exploratory because this split was previously inspected. Do not select a method using test AP."
    }
    (OUT / "run_metadata.json").write_text(json.dumps(diagnostic, indent=2))
    print("Finished. Outputs:")
    print(f"  {OUT / 'holdout_metrics_exploratory.csv'}")
    print(f"  {OUT / 'validation_cv_summary.csv'}")
    print(f"  {OUT / 'validation_cv_fold_metrics.csv'}")
    print(f"  {OUT / 'run_metadata.json'}")
    print("\nValidation CV summary (use as exploratory evidence, not proof):")
    print(summary.to_string(index=False))
    print("\nValidation-derived IQR weights:", dict(zip(MODS, np.round(iqr_w, 4))))

if __name__ == "__main__":
    main()
