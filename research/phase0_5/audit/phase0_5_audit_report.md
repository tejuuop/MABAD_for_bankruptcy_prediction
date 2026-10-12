# Phase 0.5 Methodological Audit Report

## 1. Quantile Construction
**Verified:** Quantile boundaries were correctly constructed using `qcut` on validation `disagreement_std` only, and those exact bins were applied to test using `cut`. No test labels influenced boundaries.

## 2. Positive Counts
The number of positive (bankrupt) cases in the lower quantiles is extremely small:
- Validation Q1: 1, Q2: 2, Q3: 2, Q4: 14
- Test Q1: 2, Q2: 3, Q3: 8, Q4: 34

## 3. Within-Group AP
**Methodological Assessment:** INVALID.
Computing Average Precision (AP) on subgroups containing only 1 or 2 positive cases is statistically meaningless and highly volatile. The previous claim that 'ranking performance degrades in high disagreement groups' relied heavily on comparing an AP of 0.50 (from 1 positive case) or 1.0 (from 2 positive cases) to Q4's AP. This is not robust.

## 4. Global Ranking Analysis
Instead of computing AP on tiny subgroups, evaluating overall Top-K performance provides a robust measure.
### Validation Top-K
- K_10: Precision = 0.3000, Recall = 0.1579
- K_20: Precision = 0.3000, Recall = 0.3158
- K_50: Precision = 0.1400, Recall = 0.3684
- K_100: Precision = 0.1000, Recall = 0.5263
### Test Top-K
- K_10: Precision = 0.1000, Recall = 0.0213
- K_20: Precision = 0.2000, Recall = 0.0851
- K_50: Precision = 0.1400, Recall = 0.1489
- K_100: Precision = 0.1200, Recall = 0.2553

## 5. Conditional Effect & 6. False Positive Analysis
Does disagreement provide information *beyond* d_max? There is a strong correlation between `d_max` and `disagreement_std` (Val: 0.5224, Test: 0.5794).
### False Positives in Top-K (Test)
- K_10: TP mean disagreement = 2.0179 (mean d_max=0.6229), FP mean disagreement = 2.2699 (mean d_max=0.6474)
- K_20: TP mean disagreement = 1.3866 (mean d_max=0.6140), FP mean disagreement = 2.1569 (mean d_max=0.6305)
- K_50: TP mean disagreement = 1.5306 (mean d_max=0.6007), FP mean disagreement = 2.0904 (mean d_max=0.6001)
- K_100: TP mean disagreement = 1.5542 (mean d_max=0.5851), FP mean disagreement = 2.0690 (mean d_max=0.5793)

The claim that FPs in the Top 100 have higher disagreement is technically true, but they ALSO have higher d_max on average. Disagreement is highly correlated with d_max itself.

## 7. Prevalence Confound
High disagreement is a strong predictor of bankruptcy prevalence. Thus, disagreement predicts *risk*, but because of its correlation with `d_max`, its utility as an independent *reliability* signal requires multivariate calibration.

## 8. Temporal Claim
The original claim of temporal stability is difficult to verify robustly because slicing the data by year AND by top-K leaves many years with 0 True Positives in the Top-K. (See JSON for details).

## 9. Bootstrap Validity
**Methodological Assessment:** INVALID.
The original bootstrap discarded any resampled subgroup that had 0 positive cases. Since Q1 has only 1-2 positive cases, a significant percentage of legitimate bootstrap samples (approx 36% for N=1) will have 0 positives. Filtering these out artificially inflates the AP for Q1 and makes the CI completely invalid.

## 10. Final Classification
**Classification: C = weak evidence**

The original 'A' classification was based on flawed methodology (calculating AP on N=1, discarding zero-positive bootstrap samples). The core finding—that FPs have high disagreement—is confounded by the fact that `d_max` itself is highly correlated with disagreement. The evidence that disagreement serves as an *independent reliability signal* (beyond just being correlated with the primary risk signal and baseline risk) is weak and requires a properly controlled predictive model to verify.

**Recommendation:** DO NOT proceed directly to a final reliability-aware scorer yet. We must first establish whether disagreement adds orthogonal predictive value in a multivariate model (e.g., Logistic Regression controlling for d_max) before committing to this architectural direction.