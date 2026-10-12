# Track A MVP — Static contradiction feature audit

Status: implementation prototype. Numerical severity uses training-only empirical percentiles. Narrative acknowledgement is a rules-based, unvalidated proxy.

## Data and temporal boundary

- Rows: 39,041.
- Split counts: {'train': 29752, 'test': 6381, 'val': 2908}.
- UID dates are used only for the repository's existing chronological split and display. No verified filing/publication date is available.
- Temporal contradiction score T is disabled and emitted as null.

## Feature quality

```
split  rows  positives  contradiction_available_rows  coverage_positive_rows  mean_coverage  median_coverage  max_score  min_score  liquidity_numerical_available  liquidity_narrative_available  liquidity_eligible  leverage_numerical_available  leverage_narrative_available  leverage_eligible  profitability_numerical_available  profitability_narrative_available  profitability_eligible
train 29752        126                         23228                   23228       0.619129         0.666667   0.967401        0.0                          24331                          21162               17493                         22956                         19632              15528                              24747                              26582                   22240
  val  2908         19                          2696                    2696       0.745759         1.000000   0.976897        0.0                           2867                           2062                2033                          2715                          1998               1877                               2819                               2680                    2596
 test  6381         47                          5968                    5968       0.751241         1.000000   0.860013        0.0                           6293                           4661                4590                          5929                          4347               4055                               6194                               5906                    5736
```

## Controlled MABAD comparison

Frozen held-out MABAD distances were compared on grouped validation only. No test metrics were calculated.

```
                               model                    evaluation    n  positives  prevalence  average_precision  roc_auc  precision_at_20  recall_at_20  fold
   mabad_plus_contradiction_coverage pooled_grouped_validation_oof 2908         19    0.006534           0.174541 0.942905              0.2      0.210526   NaN
     mabad_plus_severity_without_gap pooled_grouped_validation_oof 2908         19    0.006534           0.235648 0.936000              0.3      0.315789   NaN
mabad_plus_coverage_missingness_only pooled_grouped_validation_oof 2908         19    0.006534           0.207077 0.905321              0.3      0.315789   NaN
```

## Interpretation limits

- A high score means configured numerical severity exceeds the extracted acknowledgement proxy; it is not a probability and does not imply deception.
- Missing claims remain unavailable rather than being treated as reassurance.
- Auditor evidence is retained separately and is not included in management acknowledgement A.
- These outputs do not establish novelty, validity, or predictive improvement.
