# Phase 4 — pre-specified calibration robustness audit

## Purpose
This is the next experiment after Phase 3. It does not retrain MABAD and does not need `imputer.pkl`.
It compares the original maximum, raw power mean, and pre-specified label-independent calibration rules.
All learned percentile references and robust scaling values are fit only on each CV training fold.

## Run (macOS/Linux)
From this extracted `phase4_pre_registered` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn
python phase4_calibration_audit.py
```

If activation fails:
```bash
.venv/bin/python -m pip install numpy pandas scikit-learn
.venv/bin/python phase4_calibration_audit.py
```

## Outputs to send back
Zip the `outputs/` folder and upload it. Key files:
- `validation_oof_summary.csv`
- `validation_fold_metrics.csv`
- `validation_oof_predictions.csv`
- `validation_cluster_bootstrap.csv`
- `distance_scale_diagnostics.csv`
- `repeated_company_validation_rows.csv`
- `test_year_counts_audit_only.csv`
- `phase4_run_metadata.json`

## Guardrails
- Primary metric is pooled out-of-fold average precision on the 2018 validation split.
- Test labels are not scored in this phase; test data is used only to audit years and company overlap.
- Only 19 validation positives are available, so uncertainty is substantial.
- Candidate comparisons are exploratory and do not establish patent novelty.
- This script does not create a truly temporal holdout: validation contains only 2018. A genuinely future-period evaluation requires an untouched later period and a protocol fixed before examining its outcomes. The existing later test set has already been inspected and must not be treated as a fresh confirmatory test.
