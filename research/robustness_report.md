# Distance-Geometry Robustness Analysis

## Primary question
Does `distance_max_minus_min` add useful information beyond MABAD's existing `d_max` score?

## Primary test result
Frozen validation-fit logistic models on the untouched test set:

- G0 (`d_max`): AP = 0.0833, AUC = 0.8816
- G1 (`d_max + distance_max_minus_min`): AP = 0.1138, AUC = 0.8963
- AP improvement: +0.0305
- Precision@20: 20.0% -> 30.0%
- Recall@100: 25.5% -> 31.9%

Paired class-stratified bootstrap (3,000 resamples):
- G1 - G0 AP mean: +0.0337
- 95% interval: [0.0093, 0.0711]

## Validation cross-validation
5-fold stratified validation AP:
- G0: 0.2866 +/- 0.1548
- G1: 0.3024 +/- 0.1398

## Orthogonality check
Validation Pearson/Spearman correlations with range:
- d_max: -0.1476 / 0.1141
- d_mda: -0.3785 / -0.2438
- d_financial: -0.0631 / 0.1386
- d_auditor: -0.6749 / -0.8484

The strong relationship with auditor distance is an important caveat.

## Simple alternatives on test
- d_max: AP 0.0833
- d_max + financial: AP 0.0920
- d_max + MDA: AP 0.0820
- d_max + auditor: AP 0.1080
- d_max + range: AP 0.1138
- d_max + range + MDA: AP 0.1096
- d_max + range + auditor: AP 0.1138

The range model slightly beats the auditor model, but the gap is modest.

## Year-wise test
2019 (38 positives):
- G0 AP 0.1689 -> G1 AP 0.1900
- R@50 0.2895 -> 0.3158
- R@100 0.4737 -> 0.4737

2020 (9 positives):
- G0 AP 0.0732 -> G1 AP 0.1194
- R@50 0.4444 -> 0.5556
- R@100 0.5556 -> 0.6667

2021 has zero positives, so no AP/recall comparison is meaningful.

## Interpretation
Classification: B — promising but not yet sufficient to claim independent information.

The evidence supports the narrower claim that the max aggregation can be improved by incorporating the spread of modality-specific anomaly distances. It does NOT yet establish that the spread is independent of auditor anomaly information.

The next scientifically useful step is to test aggregation functions directly (max, mean, median/top-2 mean, p-norm/softmax-style aggregation), with all hyperparameters selected on validation only. That directly tests the information-loss hypothesis.
