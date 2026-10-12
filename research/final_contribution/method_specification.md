# Method Specification: Fixed Power-Mean MABAD Aggregation Audit

## Purpose

This workspace evaluates a narrow post-hoc modification to the MABAD risk-ranking aggregation. It does not retrain MABAD, modify its encoders, add a learned fusion layer, or use the Track A contradiction feature.

For each filing, the existing Phase 0 / Phase 4 artifact supplies three non-negative MABAD anomaly distances:

- \(d_F\): financial modality distance;
- \(d_M\): MD&A modality distance; and
- \(d_A\): auditor-text modality distance.

The original MABAD aggregation examined here is:

\[
M_\infty = \max(d_F,d_M,d_A).
\]

The proposed fixed alternative is the generalized (power) mean at \(p=4\):

\[
M_4 = \left(\frac{d_F^4+d_M^4+d_A^4}{3}\right)^{1/4}.
\]

For non-negative inputs, \(M_4\) lies between the arithmetic mean and the maximum. Relative to the maximum, it preserves contributions from more than one elevated modality while still weighting high distances more heavily than an arithmetic mean. This is a mathematical interpretation, not evidence that all modalities have equal reliability or that a high score is a calibrated bankruptcy probability.

## Fitting and leakage boundary

`M_4` has no learned parameter once `p=4` is fixed. The cached-prediction reproduction does not fit a scaler, imputer, classifier, threshold, or weight. It recomputes the two scores from the saved validation modality distances and compares them on the same saved rows.

The original Phase 4 candidate set was exploratory: `p=4` was one of several aggregation candidates compared on the validation set. The present audit therefore confirms computation and provenance, not independent generalization. The later test split is deliberately excluded because it was examined previously and is not a pristine holdout.

## Interpretation boundary

The method is an interpretable aggregation-rule ablation. The generalized mean is an established mathematical and ensemble-fusion family. The experiment could eventually support a MABAD-specific empirical finding only if a pre-specified version reproduces on new data or a genuinely untouched later cohort. It cannot, on its own, establish methodological novelty, causal modality reliability, or patentability.
