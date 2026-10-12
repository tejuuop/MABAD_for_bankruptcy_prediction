import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import json
import os
from pathlib import Path

# Setup paths
INPUT_FILE = "research/phase0/outputs/active_finbert_mabad_valtest_diagnostics.csv"
OUTPUT_DIR = Path("research/phase0_5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def eval_metrics(df, score_col="d_max", k_list=[10, 20, 50, 100]):
    if len(df) == 0:
        return {}
    if df["label"].sum() == 0 or df["label"].sum() == len(df):
        ap, auc = np.nan, np.nan
    else:
        ap = average_precision_score(df["label"], df[score_col])
        auc = roc_auc_score(df["label"], df[score_col])
    
    df_sorted = df.sort_values(score_col, ascending=False)
    metrics = {"ap": ap, "auc": auc}
    total_pos = df["label"].sum()
    for k in k_list:
        top_k = df_sorted.head(k)
        found_pos = top_k["label"].sum()
        recall = found_pos / total_pos if total_pos > 0 else 0
        precision = found_pos / k if k > 0 else 0
        metrics[f"recall_at_{k}"] = recall
        metrics[f"precision_at_{k}"] = precision
    return metrics

def bootstrap_ap_diff(df1, df2, score_col="d_max", n_iterations=1000):
    np.random.seed(42)
    diffs = []
    for _ in range(n_iterations):
        sample1 = df1.sample(frac=1.0, replace=True)
        sample2 = df2.sample(frac=1.0, replace=True)
        if sample1["label"].sum() > 0 and sample2["label"].sum() > 0 and sample1["label"].sum() < len(sample1) and sample2["label"].sum() < len(sample2):
            ap1 = average_precision_score(sample1["label"], sample1[score_col])
            ap2 = average_precision_score(sample2["label"], sample2[score_col])
            diffs.append(ap1 - ap2)
    if not diffs:
        return np.nan, np.nan, np.nan
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def main():
    df = pd.read_csv(INPUT_FILE)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    results = {"val": {}, "test": {}}
    
    # A. DISAGREEMENT QUANTILES
    # Use disagreement_std for quantiles
    val_df["disagreement_q"], bins = pd.qcut(val_df["disagreement_std"], q=4, retbins=True, labels=["Q1_Low", "Q2_MedLow", "Q3_MedHigh", "Q4_High"])
    bins[0] = -np.inf
    bins[-1] = np.inf
    test_df["disagreement_q"] = pd.cut(test_df["disagreement_std"], bins=bins, labels=["Q1_Low", "Q2_MedLow", "Q3_MedHigh", "Q4_High"])
    
    quantile_rows = []
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        q_results = {}
        for q in ["Q1_Low", "Q2_MedLow", "Q3_MedHigh", "Q4_High"]:
            q_df = split_df[split_df["disagreement_q"] == q]
            mets = eval_metrics(q_df)
            q_results[q] = {
                "count": int(len(q_df)),
                "positives": int(q_df["label"].sum()),
                "prevalence": float(q_df["label"].mean()) if len(q_df) > 0 else 0.0,
                "mean_d_max": float(q_df["d_max"].mean()),
                "median_d_max": float(q_df["d_max"].median()),
                "metrics": mets
            }
            quantile_rows.append({
                "split": split_name,
                "quantile": q,
                "count": len(q_df),
                "positives": q_df["label"].sum(),
                "prevalence": q_df["label"].mean(),
                "mean_d_max": q_df["d_max"].mean(),
                "median_d_max": q_df["d_max"].median(),
                "ap": mets.get("ap", np.nan),
                "auc": mets.get("auc", np.nan),
                "recall_at_50": mets.get("recall_at_50", np.nan)
            })
        results[split_name]["quantiles"] = q_results

    # B. ANALYST-BUDGET RANKING (Top-K concentration)
    # C. RELIABILITY / ERROR ANALYSIS
    # We rank all by d_max overall
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        split_df = split_df.sort_values("d_max", ascending=False).reset_index(drop=True)
        split_df["rank"] = split_df.index + 1
        split_df["is_top_50"] = split_df["rank"] <= 50
        split_df["is_top_100"] = split_df["rank"] <= 100
        
        pos_df = split_df[split_df["label"] == 1]
        neg_df = split_df[split_df["label"] == 0]
        
        # Missed positives: positives not in top 100
        missed_pos = pos_df[pos_df["rank"] > 100]
        caught_pos = pos_df[pos_df["rank"] <= 100]
        
        # False positives: negatives in top 100
        false_pos = neg_df[neg_df["rank"] <= 100]
        true_neg = neg_df[neg_df["rank"] > 100]
        
        error_stats = {
            "missed_positives_mean_disagreement": float(missed_pos["disagreement_std"].mean()) if len(missed_pos)>0 else np.nan,
            "caught_positives_mean_disagreement": float(caught_pos["disagreement_std"].mean()) if len(caught_pos)>0 else np.nan,
            "false_positives_mean_disagreement": float(false_pos["disagreement_std"].mean()) if len(false_pos)>0 else np.nan,
            "true_negatives_mean_disagreement": float(true_neg["disagreement_std"].mean()) if len(true_neg)>0 else np.nan,
            "spearman_corr_disagreement_vs_positive_rank": float(pos_df["disagreement_std"].corr(pos_df["rank"], method="spearman")) if len(pos_df)>1 else np.nan
        }
        results[split_name]["error_analysis"] = error_stats

    # D. DOMINANT MODALITY
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        dom_results = {}
        for mod in split_df["dominant_modality"].unique():
            m_df = split_df[split_df["dominant_modality"] == mod]
            mets = eval_metrics(m_df)
            dom_results[mod] = {
                "count": int(len(m_df)),
                "positives": int(m_df["label"].sum()),
                "prevalence": float(m_df["label"].mean()),
                "mean_d_max": float(m_df["d_max"].mean()),
                "metrics": mets
            }
        results[split_name]["dominant_modality"] = dom_results

    # E. TEMPORAL STABILITY
    temporal_rows = []
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        temp_res = {}
        for yr in sorted(split_df["year"].unique()):
            y_df = split_df[split_df["year"] == yr].copy()
            mets = eval_metrics(y_df)
            
            y_df = y_df.sort_values("d_max", ascending=False).reset_index(drop=True)
            y_df["rank"] = y_df.index + 1
            pos_y = y_df[y_df["label"] == 1]
            corr = float(pos_y["disagreement_std"].corr(pos_y["rank"], method="spearman")) if len(pos_y)>1 else np.nan
            
            temp_res[int(yr)] = {
                "count": int(len(y_df)),
                "positives": int(y_df["label"].sum()),
                "ap": mets.get("ap", np.nan),
                "auc": mets.get("auc", np.nan),
                "corr_disagreement_pos_rank": corr
            }
            temporal_rows.append({
                "split": split_name,
                "year": yr,
                "count": len(y_df),
                "positives": y_df["label"].sum(),
                "ap": mets.get("ap", np.nan),
                "auc": mets.get("auc", np.nan),
                "corr": corr
            })
        results[split_name]["temporal"] = temp_res

    # F. BOOTSTRAP AP DIFF (Q1 vs Q4 in Val and Test)
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        q1_df = split_df[split_df["disagreement_q"] == "Q1_Low"]
        q4_df = split_df[split_df["disagreement_q"] == "Q4_High"]
        mean_diff, ci_lower, ci_upper = bootstrap_ap_diff(q1_df, q4_df, n_iterations=500)
        results[split_name]["bootstrap_Q1_vs_Q4_AP"] = {
            "mean_diff": float(mean_diff),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        }

    # Output JSON
    with open(OUTPUT_DIR / "phase0_5_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Output CSV (Quantiles summary)
    pd.DataFrame(quantile_rows).to_csv(OUTPUT_DIR / "phase0_5_results.csv", index=False)
    
    # Generate MD Report
    report = []
    report.append("# Phase 0.5: Disagreement Reliability Analysis\n")
    report.append("## Hypothesis Tested\n*Cross-modal disagreement between financial, MD&A, and auditor anomaly scores can serve as a reliability signal for MABAD bankruptcy risk rankings.*\n")
    
    report.append("## A. Disagreement Quantiles\n")
    report.append("Validation set divided into 4 quantiles based on `disagreement_std`. Bins frozen and applied to Test.\n")
    report.append("### Validation\n")
    for q, d in results["val"]["quantiles"].items():
        report.append(f"- **{q}**: N={d['count']}, Pos={d['positives']}, Prev={d['prevalence']:.4f}, Mean d_max={d['mean_d_max']:.4f}, AP={d['metrics'].get('ap', np.nan):.4f}, Recall@100={d['metrics'].get('recall_at_100', np.nan):.4f}")
    report.append("### Test\n")
    for q, d in results["test"]["quantiles"].items():
        report.append(f"- **{q}**: N={d['count']}, Pos={d['positives']}, Prev={d['prevalence']:.4f}, Mean d_max={d['mean_d_max']:.4f}, AP={d['metrics'].get('ap', np.nan):.4f}, Recall@100={d['metrics'].get('recall_at_100', np.nan):.4f}")

    report.append("\n## C. Error Analysis\n")
    for s in ["val", "test"]:
        e = results[s]["error_analysis"]
        report.append(f"### {s.capitalize()}\n")
        report.append(f"- Missed Positives (>Rank 100) Mean Disagreement: {e['missed_positives_mean_disagreement']:.4f}")
        report.append(f"- Caught Positives (<=Rank 100) Mean Disagreement: {e['caught_positives_mean_disagreement']:.4f}")
        report.append(f"- False Positives in Top 100 Mean Disagreement: {e['false_positives_mean_disagreement']:.4f}")
        report.append(f"- Spearman Corr (Disagreement vs Positive Rank): {e['spearman_corr_disagreement_vs_positive_rank']:.4f} (Higher means high disagreement correlates with worse rank)")

    report.append("\n## D. Dominant Modality\n")
    for s in ["val", "test"]:
        report.append(f"### {s.capitalize()}\n")
        for m, d in results[s]["dominant_modality"].items():
            report.append(f"- **{m}**: N={d['count']}, Pos={d['positives']}, Prev={d['prevalence']:.4f}, AP={d['metrics'].get('ap', np.nan):.4f}")

    report.append("\n## F. Bootstrap Analysis\n")
    for s in ["val", "test"]:
        b = results[s]["bootstrap_Q1_vs_Q4_AP"]
        report.append(f"### {s.capitalize()}\n")
        report.append(f"AP Difference (Q1 - Q4): Mean={b['mean_diff']:.4f}, 95% CI=[{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")

    report.append("\n## G. Final Decision\n")
    # We will fill this via another script update or append it after reading the outputs
    
    with open(OUTPUT_DIR / "phase0_5_report.md", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    main()
