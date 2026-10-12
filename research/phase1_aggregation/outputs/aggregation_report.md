# Phase 1 — Direct Aggregation-Function Ablation

## Status and purpose

**Exploratory only.** This phase tests whether alternative mathematical aggregations of the three already-reconstructed MABAD modality distances rank bankruptcy filings better than the original maximum. The current test split was examined in earlier work, so these test results are not a pristine confirmatory evaluation.

## Data and protocol

- Validation: 2,908 filings, 19 positives (0.653%).
- Test: 6,381 filings, 47 positives (0.737%).
- Original baseline: raw `max(d_financial, d_mda, d_auditor)`, checked against saved `d_max`.
- Raw aggregations require no fitted parameters.
- Standardized aggregations use each modality mean and standard deviation fitted on validation only; the same transform is applied to test.
- Softmax/log-mean-exp temperature selected by validation AP only: `z_logmeanexp_tau_2`.
- Five-fold stratified validation CV is included as a stability check; with only 19 validation positives, fold-level AP is highly variable.
- No test labels are used for fitting or selecting temperature, but prior exploration of this test split means test comparisons remain exploratory.

## Candidates

- `raw_max_original`: original maximum aggregation.
- `raw_mean`, `raw_median`, `raw_top2_mean`, `raw_power_mean_p2`, `raw_power_mean_p4`.
- `z_max`, `z_mean`, `z_median`, `z_top2_mean`: validation-standardized modality scores.
- `rank_max`, `rank_mean`, `rank_median`, `rank_top2_mean`: modality scores transformed to empirical percentiles using validation-only reference distributions (fold-training references inside CV).
- `z_logmeanexp_tau_*`: validation-standardized log-mean-exp with a pre-declared temperature grid.

## Results

| Candidate | Validation AP | Test AP (exploratory) | Test AUC | P@20 | R@100 |
|---|---:|---:|---:|---:|---:|
| `raw_max_original` | 0.2415 | 0.0833 | 0.8816 | 20.0% | 25.5% |
| `raw_mean` | 0.2363 | 0.1587 | 0.8801 | 30.0% | 34.0% |
| `raw_median` | 0.1509 | 0.0948 | 0.8727 | 20.0% | 29.8% |
| `raw_top2_mean` | 0.2210 | 0.0979 | 0.8873 | 25.0% | 31.9% |
| `raw_power_mean_p2` | 0.2530 | 0.1494 | 0.8948 | 30.0% | 31.9% |
| `raw_power_mean_p4` | 0.3111 | 0.1254 | 0.9019 | 40.0% | 31.9% |
| `rank_max` | 0.1726 | 0.0995 | 0.9120 | 25.0% | 29.8% |
| `rank_mean` | 0.2233 | 0.1751 | 0.8521 | 40.0% | 27.7% |
| `rank_median` | 0.1803 | 0.1121 | 0.8263 | 20.0% | 29.8% |
| `rank_top2_mean` | 0.1729 | 0.1181 | 0.8741 | 25.0% | 29.8% |
| `z_max` | 0.1505 | 0.0743 | 0.9058 | 10.0% | 21.3% |
| `z_mean` | 0.2159 | 0.0991 | 0.9144 | 20.0% | 25.5% |
| `z_median` | 0.1864 | 0.1242 | 0.8311 | 35.0% | 31.9% |
| `z_top2_mean` | 0.1932 | 0.0864 | 0.9154 | 5.0% | 25.5% |
| `z_logmeanexp_tau_0.1` | 0.1529 | 0.0755 | 0.9064 | 10.0% | 21.3% |
| `z_logmeanexp_tau_0.25` | 0.1550 | 0.0778 | 0.9079 | 10.0% | 23.4% |
| `z_logmeanexp_tau_0.5` | 0.1585 | 0.0810 | 0.9108 | 10.0% | 25.5% |
| `z_logmeanexp_tau_1` | 0.1644 | 0.0875 | 0.9154 | 10.0% | 29.8% |
| `z_logmeanexp_tau_2` | 0.1810 | 0.0912 | 0.9182 | 10.0% | 23.4% |

## Validation cross-validation stability

| Candidate | Mean AP | AP SD | Mean AUC | AUC SD |
|---|---:|---:|---:|---:|
| `raw_power_mean_p4` | 0.3120 | 0.1559 | 0.9420 | 0.0371 |
| `z_mean` | 0.3014 | 0.1627 | 0.9468 | 0.0357 |
| `raw_mean` | 0.3003 | 0.1447 | 0.9108 | 0.0698 |
| `raw_top2_mean` | 0.2935 | 0.1476 | 0.9258 | 0.0692 |
| `z_median` | 0.2928 | 0.1646 | 0.8872 | 0.1109 |
| `raw_max_original` | 0.2866 | 0.1548 | 0.9465 | 0.0269 |
| `raw_power_mean_p2` | 0.2861 | 0.1592 | 0.9254 | 0.0652 |
| `z_top2_mean` | 0.2858 | 0.1401 | 0.9483 | 0.0330 |
| `rank_mean` | 0.2662 | 0.1138 | 0.8966 | 0.0420 |
| `rank_median` | 0.2603 | 0.0805 | 0.9036 | 0.0910 |
| `z_logmeanexp_tau_2` | 0.2565 | 0.1356 | 0.9504 | 0.0314 |
| `rank_top2_mean` | 0.2556 | 0.0983 | 0.9295 | 0.0590 |
| `rank_max` | 0.2538 | 0.2229 | 0.9398 | 0.0337 |
| `raw_median` | 0.2210 | 0.1107 | 0.9136 | 0.0866 |
| `z_logmeanexp_tau_1` | 0.2041 | 0.1261 | 0.9474 | 0.0342 |
| `z_logmeanexp_tau_0.5` | 0.2000 | 0.1387 | 0.9448 | 0.0360 |
| `z_logmeanexp_tau_0.25` | 0.1909 | 0.1353 | 0.9425 | 0.0351 |
| `z_logmeanexp_tau_0.1` | 0.1895 | 0.1362 | 0.9415 | 0.0353 |
| `z_max` | 0.1868 | 0.1382 | 0.9405 | 0.0358 |

## Interpretation cautions

1. Financial distance has a very different distribution from the text distances and is close to constant for most filings. Raw mean/median/power means may therefore be dominated by scale rather than meaningful modality evidence.
2. Standardization and empirical-percentile normalization reduce scale dominance but change the meaning of the original max and may be sensitive to the validation distribution. They are comparisons, not automatically better methods.
3. AP is the primary ranking metric because bankruptcy prevalence is below 1%; AUC is secondary.
4. The validation split has only 19 positives and test has 47. Apparent metric gains can be unstable.
5. These experiments cannot establish novelty. A targeted literature review and independent confirmation are still required.

## Decision rule

Do not pick a method based on test AP. Use validation CV and predeclared criteria to shortlist at most one candidate, then seek an untouched later temporal holdout or independent dataset. If that is unavailable, report this phase as exploratory and avoid confirmatory language.

## Output files

- `aggregation_metrics.csv`: validation and exploratory test metrics for all candidates.
- `validation_5fold_results.csv`: fold-level results.
- `validation_5fold_summary.csv`: mean and standard deviation across folds.
- `test_exploratory_scores.csv`: frozen per-filing candidate scores.
- `aggregation_summary.json`: machine-readable protocol and summary.
- `aggregation_ablation.py`: reproducible script.
