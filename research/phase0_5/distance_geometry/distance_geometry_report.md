# MABAD Distance-Geometry Ablation

## Research question

Does the relative geometry of financial, MD&A, and auditor anomaly distances add information beyond the original MABAD `d_max` score?

## Protocol

- Every model is fitted on validation only.
- Feature scaling is fitted on validation only.
- Test labels are never used for fitting, scaling, or selection.
- Five-fold CV is performed only inside validation as a stability check.
- Final test results are frozen holdout evaluations.

## Pre-specified models

| Model | Features | Purpose |
|---|---|---|
| G0 | d_max | Original MABAD baseline |
| G1 | d_max + max-min range | **Primary geometry test** |
| G2 | d_max + disagreement | Disagreement replication |
| G3 | d_max + range + disagreement | Combined geometry |
| G4 | three raw modality distances | Direct modality-distance test |
| G5 | d_max + dominant modality | Tests whether modality identity matters |

## Frozen test results

| Model | AP | AUC | P@20 | P@50 | P@100 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| G0_dmax | 0.0833 | 0.8816 | 0.2000 | 0.1400 | 0.1200 | 0.2553 |
| G1_dmax_plus_range | 0.1138 | 0.8963 | 0.3000 | 0.1800 | 0.1500 | 0.3191 |
| G2_dmax_plus_disagreement | 0.1065 | 0.9041 | 0.3000 | 0.1800 | 0.1500 | 0.3191 |
| G3_dmax_plus_range_plus_disagreement | 0.1140 | 0.8491 | 0.3000 | 0.1800 | 0.1500 | 0.3191 |
| G4_raw_three_distances | 0.1058 | 0.9099 | 0.2500 | 0.1800 | 0.1400 | 0.2979 |
| G5_dmax_plus_dominant_modality | 0.0828 | 0.8804 | 0.2000 | 0.1400 | 0.1200 | 0.2553 |

## Paired class-stratified bootstrap comparisons

Positive AP difference means the enhanced model beats d_max.

### G0_dmax_vs_G1_dmax_plus_range
- Mean AP difference: 0.0339
- 95% CI: [0.0095, 0.0717]
- Bootstrap P(enhanced > base): 1.0000

### G0_dmax_vs_G2_dmax_plus_disagreement
- Mean AP difference: 0.0259
- 95% CI: [0.0068, 0.0558]
- Bootstrap P(enhanced > base): 0.9994

### G0_dmax_vs_G3_dmax_plus_range_plus_disagreement
- Mean AP difference: 0.0340
- 95% CI: [0.0078, 0.0749]
- Bootstrap P(enhanced > base): 0.9988

### G0_dmax_vs_G4_raw_three_distances
- Mean AP difference: 0.0254
- 95% CI: [-0.0284, 0.0902]
- Bootstrap P(enhanced > base): 0.8078

### G0_dmax_vs_G5_dmax_plus_dominant_modality
- Mean AP difference: -0.0005
- 95% CI: [-0.0017, 0.0006]
- Bootstrap P(enhanced > base): 0.2082

## Interpretation rules

### Strong evidence
G1 improves over G0 on validation and frozen test, with a positive bootstrap interval and consistent Top-K improvement.

### Moderate evidence
G1 improves on test but the interval touches/crosses zero, or validation/test evidence is inconsistent. Treat as a lead only.

### Redundant/negative
G1 does not consistently beat G0. Do not claim that distance geometry is an incremental contribution.

### Important caution
G4/G5 are secondary ablations. They must not be used to cherry-pick a winning feature combination after seeing test results.
