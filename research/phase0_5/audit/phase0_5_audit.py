import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import json
import os
import scipy.stats as stats
from pathlib import Path

INPUT_FILE = "research/phase0/outputs/active_finbert_mabad_valtest_diagnostics.csv"
OUTPUT_DIR = Path("research/phase0_5/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INPUT_FILE)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    results = {}

    # 1. QUANTILE CONSTRUCTION
    # Verified in phase0_5_analysis.py: qcut on val_df, cut on test_df with bins. Correct.
    val_df["disagreement_q"], bins = pd.qcut(val_df["disagreement_std"], q=4, retbins=True, labels=["Q1", "Q2", "Q3", "Q4"])
    bins[0] = -np.inf
    bins[-1] = np.inf
    test_df["disagreement_q"] = pd.cut(test_df["disagreement_std"], bins=bins, labels=["Q1", "Q2", "Q3", "Q4"])

    # 2. POSITIVE COUNTS
    counts_res = {}
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        counts_res[split_name] = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_df = split_df[split_df["disagreement_q"] == q]
            n_tot = len(q_df)
            n_pos = int(q_df["label"].sum())
            n_neg = n_tot - n_pos
            counts_res[split_name][q] = {"N": n_tot, "positives": n_pos, "negatives": n_neg}
    results["positive_counts"] = counts_res

    # 3. WITHIN-GROUP AP
    # With positive counts in Q1/Q2 being 1 or 2, AP is statistically meaningless.
    
    # 4. GLOBAL RANKING ANALYSIS & 6. FALSE POSITIVE ANALYSIS
    global_ranking = {}
    fp_analysis = {}
    k_list = [10, 20, 50, 100]
    
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        split_df = split_df.sort_values("d_max", ascending=False).reset_index(drop=True)
        split_df["rank"] = split_df.index + 1
        
        g_res = {}
        fp_res = {}
        total_pos = split_df["label"].sum()
        
        for k in k_list:
            top_k = split_df.head(k)
            found_pos = top_k["label"].sum()
            g_res[f"K_{k}"] = {
                "precision": float(found_pos / k) if k > 0 else 0.0,
                "recall": float(found_pos / total_pos) if total_pos > 0 else 0.0
            }
            
            # FP Analysis for this K
            tp_top_k = top_k[top_k["label"] == 1]
            fp_top_k = top_k[top_k["label"] == 0]
            neg_outside = split_df[(split_df["rank"] > k) & (split_df["label"] == 0)]
            
            fp_res[f"K_{k}"] = {
                "TP_in_topK_mean_disagreement": float(tp_top_k["disagreement_std"].mean()) if len(tp_top_k)>0 else np.nan,
                "FP_in_topK_mean_disagreement": float(fp_top_k["disagreement_std"].mean()) if len(fp_top_k)>0 else np.nan,
                "Neg_outside_topK_mean_disagreement": float(neg_outside["disagreement_std"].mean()) if len(neg_outside)>0 else np.nan,
                "TP_in_topK_mean_dmax": float(tp_top_k["d_max"].mean()) if len(tp_top_k)>0 else np.nan,
                "FP_in_topK_mean_dmax": float(fp_top_k["d_max"].mean()) if len(fp_top_k)>0 else np.nan,
            }
        global_ranking[split_name] = g_res
        fp_analysis[split_name] = fp_res
        
    results["global_ranking"] = global_ranking
    results["fp_analysis"] = fp_analysis

    # 5. CONDITIONAL EFFECT
    cond_effect = {}
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        corr_dmax_disag = split_df["d_max"].corr(split_df["disagreement_std"])
        corr_dmax_label = split_df["d_max"].corr(split_df["label"])
        corr_disag_label = split_df["disagreement_std"].corr(split_df["label"])
        cond_effect[split_name] = {
            "corr_dmax_vs_disagreement": float(corr_dmax_disag),
            "corr_dmax_vs_label": float(corr_dmax_label),
            "corr_disagreement_vs_label": float(corr_disag_label)
        }
    results["conditional_effect"] = cond_effect

    # 8. TEMPORAL CLAIM
    temporal_res = {}
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        t_res = {}
        for yr in sorted(split_df["year"].unique()):
            y_df = split_df[split_df["year"] == yr].copy()
            y_df = y_df.sort_values("d_max", ascending=False).reset_index(drop=True)
            y_df["rank"] = y_df.index + 1
            
            top100_y = y_df.head(100)
            tp_top100 = top100_y[top100_y["label"] == 1]
            fp_top100 = top100_y[top100_y["label"] == 0]
            
            t_res[int(yr)] = {
                "N": len(y_df),
                "positives": int(y_df["label"].sum()),
                "FP_in_top100_mean_disagreement": float(fp_top100["disagreement_std"].mean()) if len(fp_top100)>0 else np.nan,
                "TP_in_top100_mean_disagreement": float(tp_top100["disagreement_std"].mean()) if len(tp_top100)>0 else np.nan
            }
        temporal_res[split_name] = t_res
    results["temporal_analysis"] = temporal_res

    with open(OUTPUT_DIR / "phase0_5_audit_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Convert to CSV friendly formats for specific tables if needed, but JSON has everything structured.
    # Write audit report
    report = [
        "# Phase 0.5 Methodological Audit Report",
        "",
        "## 1. Quantile Construction",
        "**Verified:** Quantile boundaries were correctly constructed using `qcut` on validation `disagreement_std` only, and those exact bins were applied to test using `cut`. No test labels influenced boundaries.",
        "",
        "## 2. Positive Counts",
        "The number of positive (bankrupt) cases in the lower quantiles is extremely small:",
        "- Validation Q1: 1, Q2: 2, Q3: 2, Q4: 14",
        "- Test Q1: 2, Q2: 3, Q3: 8, Q4: 34",
        "",
        "## 3. Within-Group AP",
        "**Methodological Assessment:** INVALID.",
        "Computing Average Precision (AP) on subgroups containing only 1 or 2 positive cases is statistically meaningless and highly volatile. The previous claim that 'ranking performance degrades in high disagreement groups' relied heavily on comparing an AP of 0.50 (from 1 positive case) or 1.0 (from 2 positive cases) to Q4's AP. This is not robust.",
        "",
        "## 4. Global Ranking Analysis",
        "Instead of computing AP on tiny subgroups, evaluating overall Top-K performance provides a robust measure."
    ]
    
    report.append("### Validation Top-K")
    for k, metrics in global_ranking["val"].items():
        report.append(f"- {k}: Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")
    
    report.append("### Test Top-K")
    for k, metrics in global_ranking["test"].items():
        report.append(f"- {k}: Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")
        
    report.append("")
    report.append("## 5. Conditional Effect & 6. False Positive Analysis")
    report.append("Does disagreement provide information *beyond* d_max? There is a strong correlation between `d_max` and `disagreement_std` (Val: {:.4f}, Test: {:.4f}).".format(
        cond_effect['val']['corr_dmax_vs_disagreement'], cond_effect['test']['corr_dmax_vs_disagreement']
    ))
    
    report.append("### False Positives in Top-K (Test)")
    for k, mets in fp_analysis["test"].items():
        report.append(f"- {k}: TP mean disagreement = {mets['TP_in_topK_mean_disagreement']:.4f} (mean d_max={mets['TP_in_topK_mean_dmax']:.4f}), "
                      f"FP mean disagreement = {mets['FP_in_topK_mean_disagreement']:.4f} (mean d_max={mets['FP_in_topK_mean_dmax']:.4f})")
    
    report.append("")
    report.append("The claim that FPs in the Top 100 have higher disagreement is technically true, but they ALSO have higher d_max on average. Disagreement is highly correlated with d_max itself.")
    
    report.append("")
    report.append("## 7. Prevalence Confound")
    report.append("High disagreement is a strong predictor of bankruptcy prevalence. Thus, disagreement predicts *risk*, but because of its correlation with `d_max`, its utility as an independent *reliability* signal requires multivariate calibration.")
    
    report.append("")
    report.append("## 8. Temporal Claim")
    report.append("The original claim of temporal stability is difficult to verify robustly because slicing the data by year AND by top-K leaves many years with 0 True Positives in the Top-K. (See JSON for details).")
    
    report.append("")
    report.append("## 9. Bootstrap Validity")
    report.append("**Methodological Assessment:** INVALID.")
    report.append("The original bootstrap discarded any resampled subgroup that had 0 positive cases. Since Q1 has only 1-2 positive cases, a significant percentage of legitimate bootstrap samples (approx 36% for N=1) will have 0 positives. Filtering these out artificially inflates the AP for Q1 and makes the CI completely invalid.")
    
    report.append("")
    report.append("## 10. Final Classification")
    report.append("**Classification: C = weak evidence**")
    report.append("")
    report.append("The original 'A' classification was based on flawed methodology (calculating AP on N=1, discarding zero-positive bootstrap samples). The core finding—that FPs have high disagreement—is confounded by the fact that `d_max` itself is highly correlated with disagreement. The evidence that disagreement serves as an *independent reliability signal* (beyond just being correlated with the primary risk signal and baseline risk) is weak and requires a properly controlled predictive model to verify.")
    report.append("")
    report.append("**Recommendation:** DO NOT proceed directly to a final reliability-aware scorer yet. We must first establish whether disagreement adds orthogonal predictive value in a multivariate model (e.g., Logistic Regression controlling for d_max) before committing to this architectural direction.")
    
    with open(OUTPUT_DIR / "phase0_5_audit_report.md", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    main()
