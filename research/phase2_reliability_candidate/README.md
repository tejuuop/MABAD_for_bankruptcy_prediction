# Phase 2 candidate experiment: reliability-weighted aggregation

## Purpose
This adds a small, reproducible post-hoc experiment on the existing Phase 0.5 modality-distance CSV. It does not retrain MABAD and does not need `imputer.pkl`.

## Hypothesis
If modality distances have materially different scales and dispersion, a scale-aware aggregation may rank rare bankruptcy cases more effectively than the original raw maximum. The candidate converts each modality distance to an empirical percentile using only the validation/fold-training distribution, then compares:
- original raw maximum
- raw mean
- equal-weight percentile mean
- percentile mean weighted by each modality's validation IQR
- a pre-specified sensitivity ablation that damps the low-variation financial modality

The IQR weighting is a proxy for variation, **not a proven measure of reliability**. Results may disconfirm the hypothesis. Do not describe this as established novelty or a patentable invention without prior-art review.

## Run locally
Use Python 3.10–3.12 if possible. From this folder:

```bash
python -m venv .venv
# macOS/Linux (bash/zsh):
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn
python phase2_reliability_candidate.py
```

The input CSV is already included under `inputs/`. No network access, original model training, or `imputer.pkl` is required.

## Outputs
- `outputs/validation_cv_summary.csv`: main comparison to inspect first
- `outputs/validation_cv_fold_metrics.csv`: per-fold scores and weights
- `outputs/holdout_metrics_exploratory.csv`: validation/test metrics; test is explicitly exploratory
- `outputs/run_metadata.json`: row counts, IQRs, and weights

## Evaluation rules
1. Treat average precision (AP) as the primary metric because positive cases are rare; report ROC-AUC as secondary.
2. Report mean and SD across the 5 validation folds, not just the best fold.
3. Do not tune weights repeatedly against the test split.
4. Since the validation set has very few positive cases, fold-level AP is noisy.
5. A credible next step is a genuinely untouched later-period holdout or an independent dataset. If none is available before submission, clearly label findings exploratory.
