# Minimum Decisive Experiment Plan

## Decision

The selected candidate is **fixed generalized-mean aggregation of the three frozen MABAD modality distances**, with the pre-existing raw power mean at `p = 4` as the proposed score and the original raw maximum as the baseline.

This selection is provisional and evidence-driven. Phase 4 provides the only directly comparable saved row-level validation predictions. The contradiction prototype is excluded from this experiment because its narrative acknowledgement labels are unvalidated and its saved comparison is not a common-scoring-protocol comparison. Reliability-aware and geometry candidates are excluded because their available results are test-informed or confounded and do not establish incremental reliability value.

## Primary hypothesis

On the existing 2018 validation observations, the fixed, label-independent score

\[
  M_4(d) = \left((d_F^4+d_M^4+d_A^4)/3\right)^{1/4}
\]

has higher pooled out-of-fold average precision than the original score

\[
  M_\infty(d) = \max(d_F,d_M,d_A).
\]

The power `p = 4` is already selected by the Phase 4 exploratory candidate set. This audit does not re-select it and does not claim that its selection was independent of the validation observations.

## Data and comparison protocol

- Source: cached Phase 4 validation OOF predictions only.
- Rows: the 2,908 `split == val` rows in the saved file.
- Positives: 19.
- Groups: company ID; the saved `oof_fold` is retained for fold reporting.
- Baseline and proposal use the exact same rows, labels, company IDs, and folds.
- No model is fit. Both scores are recomputed directly from the three frozen modality distances and checked against the saved OOF score columns.
- No test labels or test metrics are read, scored, tuned, or reported.

## Metrics

- Primary: pooled average precision (AP).
- Secondary: ROC-AUC; recall and precision at review budgets 20, 50, and 100.
- Variability: fold metrics using the saved company-grouped OOF fold assignment.
- Uncertainty: paired, company-cluster bootstrap for the AP difference, stratified by whether a company has a positive validation row.

## Ablation

The sole planned ablation is the original maximum versus the fixed power mean. The method is intentionally limited to isolate the aggregation rule. No learned reliability weights, normalization rule, contradiction feature, or second-stage classifier is added.

## Stopping rule

This audit supports only a reproducibility finding if all of the following hold:

1. score recomputation matches the saved OOF scores to numerical tolerance;
2. baseline and proposal use identical UIDs, labels, groups, and folds;
3. the pre-existing AP difference and bootstrap interval are reproduced; and
4. all implementation checks pass.

It does **not** support a predictive-improvement claim unless a later untouched or independently sourced evaluation confirms a pre-specified rule. If the company-cluster interval includes zero, the validation evidence is recorded as promising but statistically uncertain, not as superiority.

## Expected failure conditions

- Missing or duplicate UIDs, label mismatch, or a score discrepancy blocks reproduction.
- A changed fold assignment blocks fold-level reproduction.
- A bootstrap interval including zero prevents a confirmed improvement claim.
- Any use of later test outcomes for selection invalidates confirmatory interpretation; the script is designed not to load those outcomes.
