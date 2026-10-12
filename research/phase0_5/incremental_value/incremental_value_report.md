# Controlled Incremental-Value Analysis

## Research question

Does cross-modal disagreement provide predictive information beyond the existing MABAD `d_max` anomaly score?

## Protocol

- Model parameters and feature scaling were fitted on validation only.
- Test labels were not used for fitting, scaling, or model selection.
- Final test predictions are therefore frozen holdout predictions.
- A 5-fold stratified CV on validation is included only as a stability check.

## Models

- **M0:** `d_max`
- **M1:** `disagreement_std`
- **M2:** `d_max + disagreement_std`
- **M3:** `d_max + disagreement_std + distance_max_minus_min`

## Final test results

| Model | AP | AUC | P@10 | P@20 | P@50 | P@100 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0_dmax | 0.0833 | 0.8816 | 0.1000 | 0.2000 | 0.1400 | 0.1200 | 0.0213 | 0.0851 | 0.1489 | 0.2553 |
| M1_disagreement | 0.0538 | 0.7860 | 0.1000 | 0.0500 | 0.0400 | 0.0600 | 0.0213 | 0.0213 | 0.0426 | 0.1277 |
| M2_dmax_plus_disagreement | 0.0743 | 0.8894 | 0.1000 | 0.1000 | 0.1400 | 0.1100 | 0.0213 | 0.0426 | 0.1489 | 0.2340 |
| M3_dmax_plus_disagreement_plus_range | 0.1025 | 0.9074 | 0.1000 | 0.3000 | 0.1600 | 0.1300 | 0.0213 | 0.1277 | 0.1702 | 0.2766 |

## Incremental test comparisons

### M0_dmax_vs_M2_dmax_plus_disagreement
- Mean AP difference (enhanced - base): -0.0099
- 95% class-stratified bootstrap CI: [-0.0269, 0.0018]

### M0_dmax_vs_M3_dmax_plus_disagreement_plus_range
- Mean AP difference (enhanced - base): 0.0221
- 95% class-stratified bootstrap CI: [0.0024, 0.0552]

### M1_disagreement_vs_M2_dmax_plus_disagreement
- Mean AP difference (enhanced - base): 0.0262
- 95% class-stratified bootstrap CI: [-0.0404, 0.0872]

## Interpretation rule

The key comparison is **M0 vs M2**.

- If M2 consistently improves AP/Top-K performance over M0 and the bootstrap interval for AP improvement is clearly above zero, this supports incremental information from disagreement.
- If M2 is nearly identical to M0, disagreement is probably redundant with the existing anomaly score.
- If M2 improves validation but not frozen test, treat the result as unstable rather than claiming a contribution.
- M3 is an ablation showing whether adding the simple max-min distance range explains the apparent disagreement effect.

## Important limitation

This experiment establishes predictive incremental value (or lack of it). It does not by itself establish that disagreement is a causal or mechanistically independent 'reliability' signal.
