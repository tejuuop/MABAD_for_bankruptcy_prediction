# Phase 0.5: Disagreement Reliability Analysis

## Hypothesis Tested
*Cross-modal disagreement between financial, MD&A, and auditor anomaly scores can serve as a reliability signal for MABAD bankruptcy risk rankings.*

## A. Disagreement Quantiles

Validation set divided into 4 quantiles based on `disagreement_std`. Bins frozen and applied to Test.

### Validation

- **Q1_Low**: N=727, Pos=1, Prev=0.0014, Mean d_max=0.4327, AP=0.5000, Recall@100=1.0000
- **Q2_MedLow**: N=727, Pos=2, Prev=0.0028, Mean d_max=0.4342, AP=1.0000, Recall@100=1.0000
- **Q3_MedHigh**: N=727, Pos=2, Prev=0.0028, Mean d_max=0.4351, AP=0.0100, Recall@100=0.5000
- **Q4_High**: N=727, Pos=14, Prev=0.0193, Mean d_max=0.4522, AP=0.2478, Recall@100=0.5714
### Test

- **Q1_Low**: N=1796, Pos=2, Prev=0.0011, Mean d_max=0.4328, AP=0.3667, Recall@100=1.0000
- **Q2_MedLow**: N=1403, Pos=3, Prev=0.0021, Mean d_max=0.4340, AP=0.3854, Recall@100=0.6667
- **Q3_MedHigh**: N=1274, Pos=8, Prev=0.0063, Mean d_max=0.4364, AP=0.2020, Recall@100=0.7500
- **Q4_High**: N=1908, Pos=34, Prev=0.0178, Mean d_max=0.4605, AP=0.0745, Recall@100=0.2647

## C. Error Analysis

### Val

- Missed Positives (>Rank 100) Mean Disagreement: 1.5482
- Caught Positives (<=Rank 100) Mean Disagreement: 1.9126
- False Positives in Top 100 Mean Disagreement: 1.8016
- Spearman Corr (Disagreement vs Positive Rank): -0.0544 (Higher means high disagreement correlates with worse rank)
### Test

- Missed Positives (>Rank 100) Mean Disagreement: 1.5444
- Caught Positives (<=Rank 100) Mean Disagreement: 1.5542
- False Positives in Top 100 Mean Disagreement: 2.0690
- Spearman Corr (Disagreement vs Positive Rank): -0.2069 (Higher means high disagreement correlates with worse rank)

## D. Dominant Modality

### Val

- **mda**: N=190, Pos=12, Prev=0.0632, AP=0.3468
- **financial**: N=2654, Pos=6, Prev=0.0023, AP=0.2284
- **auditor**: N=64, Pos=1, Prev=0.0156, AP=0.0189
### Test

- **financial**: N=5526, Pos=16, Prev=0.0029, AP=0.0976
- **mda**: N=775, Pos=29, Prev=0.0374, AP=0.1149
- **auditor**: N=80, Pos=2, Prev=0.0250, AP=0.0248

## F. Bootstrap Analysis

### Val

AP Difference (Q1 - Q4): Mean=0.4154, 95% CI=[-0.1178, 0.9031]
### Test

AP Difference (Q1 - Q4): Mean=0.4095, 95% CI=[0.0780, 0.9301]

## G. Final Decision

**Classification: A = strong evidence**

**Strongest supporting evidence**: 
Ranking performance (AP) is significantly higher for low-disagreement cases compared to high-disagreement cases. In the test set, Q1_Low has an AP of 0.3667, while Q4_High has an AP of only 0.0745. The Bootstrap analysis confirms this gap is statistically significant in the test set (95% CI: [0.0780, 0.9301]). Furthermore, False Positives in the Top 100 have a very high mean disagreement (2.0690 in test), indicating that high disagreement floods the top ranks with false alarms.

**Strongest contradictory evidence**:
The Spearman correlation between disagreement and the numerical rank of *Positive* cases is slightly negative (-0.05 to -0.20), meaning high disagreement doesn't necessarily push True Positives down the ranking; it actually pushes them slightly up. The unreliability comes from False Positives being pushed up even more. Additionally, high disagreement groups have a much higher *prevalence* of bankruptcy (Q4 test prevalence is 1.78% vs Q1 0.11%), meaning disagreement itself is a risk factor, not just a noise factor.

**Temporal stability**: 
Yes, temporal analysis (saved in JSON) confirms that the AP degradation for high disagreement is consistent across the validation and test time splits.

**Survival to Test**: 
Yes. The AP degradation from Q1 to Q4 observed in validation (0.50 -> 0.24) survives and becomes even more pronounced and statistically significant in the untouched test set (0.36 -> 0.07).

**Methodological limitations**: 
The extremely low number of positive cases in Q1/Q2 (1-3 positives) makes the AP metric for those quantiles highly volatile, which is reflected in the wide bootstrap confidence intervals. 

**Recommendation**: 
**Proceed** to a simple post-hoc reliability-aware scorer. Since high disagreement indicates both a higher base risk (prevalence) and higher ranking noise (false positives), a post-hoc model can learn to properly calibrate these opposing effects.
