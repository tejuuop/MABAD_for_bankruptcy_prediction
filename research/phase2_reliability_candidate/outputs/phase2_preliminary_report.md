# Phase 2 candidate run — preliminary results

## What was run
Post-hoc experiment on saved Phase 0.5 modality distances. No MABAD retraining and no `imputer.pkl` required.

## Key diagnostic
- Validation rows: 2908; positives: 19.
- Test rows: 6381; positives: 47.
- Validation IQRs:
  - `d_financial`: 0.002968
  - `d_mda`: 0.085698
  - `d_auditor`: 0.123255

The financial-distance IQR is much smaller than the two text-distance IQRs. This supports testing scale-aware aggregation, but low variance alone does not prove low reliability.

## Five-fold validation results

| Method | Mean AP | AP SD | Mean ROC-AUC | AUC SD |
|---|---:|---:|---:|---:|
| raw_mean | 0.3425 | 0.3897 | 0.9113 | 0.0809 |
| rank_financial_damped_0.1_1_1 | 0.3313 | 0.3403 | 0.8528 | 0.1055 |
| rank_iqr_weighted | 0.3293 | 0.3603 | 0.8128 | 0.1315 |
| rank_equal_mean | 0.3068 | 0.2551 | 0.8956 | 0.0892 |
| original_raw_max | 0.2878 | 0.2668 | 0.9491 | 0.0436 |

## Preliminary interpretation
- In this run, raw mean has the highest mean AP across the five validation folds (about 0.342), followed by the fixed financial-damped percentile score (about 0.331) and IQR-weighted percentile score (about 0.329).
- The original raw maximum has mean AP about 0.288. However, AP standard deviations are large (roughly 0.27–0.39), reflecting very few positive cases per fold. Do not interpret these small-sample estimates as a stable win.
- The IQR-weighted score's mean AUC is relatively low in this run; AP and AUC rankings differ. This warns against declaring a winner based on one metric.
- The existing test split has been inspected in prior work. Its results are marked exploratory and must not be used to select the method.

## What to do next
1. Run the script on your machine and keep the generated outputs unchanged.
2. Inspect `validation_cv_summary.csv` and `validation_cv_fold_metrics.csv`; check that counts and results match.
3. Freeze the hypotheses before any further evaluation. Do not keep tuning on the test set.
4. If you have time, seek a truly untouched later-period or independent dataset for confirmation. If unavailable, present the work as exploratory.
5. For the invention disclosure, describe this as a candidate post-hoc aggregation method and state that novelty/patentability is unverified pending prior-art review.

## Reproducibility warning
The validation set has 19 positives total, so a five-fold split has only a few positive examples per fold. Report mean and standard deviation, show the individual fold results, and avoid claims of statistical certainty. These are preliminary exploratory results, not proof of a novel or patentable invention.
