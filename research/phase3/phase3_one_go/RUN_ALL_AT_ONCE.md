# Phase 3: one-run audit and exploration

## Purpose
This is the next step after Phase 0.5 / Phase 1 / Phase 2. It audits score distributions and runs a leakage-aware comparison of aggregation candidates using the saved distance CSV. It does not retrain MABAD and does not need `imputer.pkl`.

## Run
Open a terminal in this extracted `phase3_one_go` folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn
python phase3_audit_exploration.py
```

If shell activation fails, run directly:

```bash
.venv/bin/python -m pip install numpy pandas scikit-learn
.venv/bin/python phase3_audit_exploration.py
```

On Windows, use `.venv\\Scripts\\python.exe` instead of `.venv/bin/python`.

## Upload these results back
Zip the whole `outputs` folder or upload these files individually:
1. `validation_oof_summary.csv` — most important: pooled out-of-fold validation comparison.
2. `validation_fold_metrics.csv` — per-fold results.
3. `validation_oof_paired_bootstrap.csv` — uncertainty around AP differences vs original maximum.
4. `fold_fit_diagnostics.csv` — fold counts and fold-specific IQR weights.
5. `modality_distance_diagnostics.csv` and `validation_spearman_correlations.csv` — scale and correlation diagnostics.
6. `test_exploratory_all_methods.csv` — include for transparency only; do not select a method using it.
7. `audit_run_metadata.json` and `availability_subgroups.csv`.
8. `validation_oof_predictions.csv` — include if file size is manageable; useful for independent checking.

## Methodology safeguards
- All percentile references and IQR weights for each validation fold are fitted on that fold's training partition only.
- The aggregation weights are not fitted using labels.
- Pooled OOF validation AP is the primary metric. Per-fold AP is shown but will be unstable because each fold has only a few positives.
- The test split covers a later period but was already inspected in prior experiments; its metrics are exploratory only. Do not choose a method using test performance.
- This is research exploration, not evidence of patentability by itself.
