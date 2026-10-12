# Research History and Current Direction — MABAD Bankruptcy Prediction

**Last updated:** 2026-10-09  
**Purpose:** Project context, decisions, evidence, caveats, and next experiment. Share this with Gemini alongside the `research/` archive.

## 1. Overall purpose

We are working on a research project based on MABAD (a multimodal anomaly-based approach for bankruptcy prediction). The data combines financial/XBRL information with text from MD&A sections and auditor opinions. Bankruptcy is extremely rare, and the financial representation is highly incomplete.

Our goal is **not** to reimplement the original paper from scratch or to stack arbitrary features until a metric improves. We want to identify a specific, testable limitation in the baseline, investigate it using the saved artifacts, and—only if supported by rigorous evidence—develop a defensible novel research contribution.

The current working hypothesis is:

> **Hard maximum aggregation of modality-specific anomaly distances may discard useful information about the joint configuration of financial, MD&A, and auditor distances. A better aggregation rule or carefully justified refinement may improve bankruptcy-risk ranking.**

This is a hypothesis, not yet a validated contribution.

## 2. Dataset and evaluation

The existing audit reported approximately:
- 39,041 synchronized firm-filings.
- 192 bankruptcy-positive filings overall (~0.492% prevalence).
- 4,855 companies.
- Dates from 1999-12-31 to 2021-10-31.
- ~9,700 possible financial/XBRL fields, with around 208 observed per filing on average (~97.85% missingness).
- MD&A and auditor text are available in the current data files.

Reconstructed chronological split:

| Split | Rows | Positives | Period |
|---|---:|---:|---|
| Train | 29,752 | 126 | through 2017-12-31 |
| Validation | 2,908 | 19 | calendar year 2018 |
| Test | 6,381 | 47 | 2019-01-01 to 2021-10-31 |

The saved Phase 0 distance file uses split labels `val` and `test`. Check exact files for row-count differences before quoting numbers.

With only 47 positives in test, ROC-AUC alone is inadequate. We focus on Average Precision (AP / PR-AUC), precision/recall at review budgets (top 10, 20, 50, 100), uncertainty, and stability. Metrics are noisy under extreme imbalance.

## 3. Why we are not reimplementing the base paper

The reproduction attempt stalled because code expected an `imputer.pkl` that was not generated. Rather than spend limited time blindly debugging the full reproduction, we audited the repository and found useful saved artifacts:
- MABAD model weights and an XGBoost model.
- Normality centers (`c_tab`, `c_mda`, `c_aud`) in `tri_sad.pt`.
- TF-IDF vectorizers.
- FinBERT embeddings for MD&A and auditor text.
- Historical aggregate baseline metrics.
- No saved per-example final predictions from the original runs; validation/test predictions and modality distances had to be reconstructed through inference.

There are separate artifact chains:
- `models/saved_models/multisource/MABAD/tri_sad.pt` is FinBERT-shaped (text input dimension 768).
- `models/saved_models/tfidf_weights` is TF-IDF-shaped (text input dimension 1000).

**Do not mix artifact chains.** The active Phase 0 reconstruction used the FinBERT MABAD artifact and matching embeddings. Its results were broadly consistent with historical `MABAD_FinBERT_FullDoc` results (AUC around 0.90 and AP around 0.10). This supports using the saved artifact for controlled diagnostic work, but does not prove every detail of the original paper was reproduced.

## 4. What we have done and why

### Phase 0 — Dataset and artifact audit

**Purpose:** Determine what data/models are available, recover a traceable inference path, and identify plausible research directions.

Work performed:
- Audited financial-feature completeness and text availability.
- Reconstructed the chronological split.
- Loaded the active FinBERT MABAD artifact and matching embeddings.
- Reconstructed per-modality anomaly distances: `d_financial`, `d_mda`, `d_auditor`.
- Derived `d_max`, dominant modality, pairwise gaps, `distance_max_minus_min`, and related diagnostics.
- Saved validation/test outputs and reports under `research/phase0/outputs/`.

Reported Phase 0 metrics:
- `d_max`: AP 0.0989, AUC 0.9003, Recall@50 0.1364, Recall@100 0.2424.
- `d_mda`: AP 0.0951, AUC 0.8796.

Findings:
- Financial missingness alone looked like a weak ranking signal.
- MD&A distance was a useful individual modality score.
- Max of the three distances ranked risk reasonably well.
- Initial disagreement/error associations could be confounded by overall anomaly magnitude.

Why this matters: Phase 0 replaced assumptions with an auditable analysis of saved artifacts and produced the modality-level distances needed to test the aggregation mechanism.

### Phase 0.5 — Cross-modal disagreement hypothesis

Initial idea: when modality distances disagree, the overall MABAD risk score may be less reliable.

An exploratory analysis used validation-derived quantile boundaries and applied them to test. It initially looked as though low-disagreement cases ranked much better than high-disagreement cases.

An audit found major limitations:
- AP within quantile groups was unstable because groups contained very few positives.
- The bootstrap discarded zero-positive resamples, biasing uncertainty estimates.
- `disagreement_std` correlated with `d_max` on test (Pearson r about 0.579), so disagreement may proxy overall anomaly magnitude.
- Temporal slices had few or no positives.

**Conclusion:** The claim that disagreement is an independent reliability signal is not established. Do not proceed directly to a scorer based on this hypothesis.

### Incremental-value test

We compared validation-fit logistic models using `d_max` alone versus `d_max` plus disagreement-related features, freezing the model before test.

Reported results from the saved incremental experiment:
- `d_max`: test AP 0.0833, AUC 0.8816, Precision@20 20%, Recall@100 25.5%.
- `d_max + disagreement`: AP 0.0743, AUC 0.8894, Precision@20 10%, Recall@100 23.4%.
- `d_max + distance_max_minus_min`: AP about 0.1025 in that experiment.

A later direct robustness rerun using the supplied modality-distance CSV reported:
- `d_max`: AP 0.0833, AUC 0.8816.
- `d_max + distance_max_minus_min` (range): AP 0.1138, AUC 0.8963.
- Precision@20: 20% for `d_max`, 30% for `d_max + range`.
- Recall@100: 25.5% for `d_max`, 31.9% for `d_max + range`.

The direct rerun's paired class-stratified bootstrap estimated mean AP improvement around +0.0337, 95% CI [+0.0093, +0.0711]. This is conditional on fixed predictions and this test set; it does not capture all model-selection uncertainty.

Validation 5-fold results were only modestly favorable:
- `d_max`: mean AP 0.2866 (fold SD 0.1548).
- `d_max + range`: mean AP 0.3024 (fold SD 0.1398).

Major caveat: the range feature is strongly related to auditor distance (validation Spearman correlation about -0.848), and its validation logistic coefficient was suggestive but not conventionally significant (p about 0.084). These results are promising but not enough to claim an independent signal or generalizable method. The two reported range AP values (0.1025 and 0.1138) came from different runs/procedures; do not combine them without checking the exact scripts and outputs.

Current interpretation: the useful research lead is not “disagreement predicts unreliability.” It is that **hard max aggregation may discard information about the joint configuration of the modality distances**. The range feature may recover some information, but stronger controlled tests are needed.

## 5. Current evidence snapshot

Latest direct test comparison:

| Score/model | Test AP | Test ROC-AUC | Precision@20 | Recall@100 |
|---|---:|---:|---:|---:|
| `d_max` | 0.0833 | 0.8816 | 20% | 25.5% |
| `d_max + range` | 0.1138 | 0.8963 | 30% | 31.9% |
| `d_max + d_mda` | 0.0820 | 0.8842 | — | — |
| `d_max + d_auditor` | about 0.1080 | about 0.8888 | verify exact output | verify exact output |

Treat these as exploratory fixed-test comparisons, not final confirmatory results. The test set has only 47 positives, and earlier exploration has influenced the current hypothesis.

Other diagnostics:
- Top-100 rankings from `d_max` and `d_max + range` overlap in 81/100 cases.
- The range model brought four positives into its top 100 that the baseline did not; one baseline top-100 positive fell out.
- This suggests refinement rather than wholesale ranking changes.
- Year-wise results were directionally favorable in 2019 and 2020, but 2020 has only 9 positives and 2021 has none. These slices do not establish temporal robustness.

## 6. Candidate research question and contribution

Candidate question:

> Does replacing MABAD's hard maximum aggregation of modality-specific anomaly distances with a principled aggregation function improve bankruptcy-risk ranking under extreme class imbalance?

Potential contribution, **only if validated**: a controlled study of modality-distance aggregation for multimodal anomaly-based bankruptcy prediction, demonstrating a consistent, interpretable and reproducible improvement over hard max without leakage.

Why this direction is promising:
- It tests a specific baseline mechanism rather than adding arbitrary features.
- It uses saved MABAD artifacts and reconstructed distances, avoiding full reimplementation.
- It is interpretable: aggregators encode different assumptions about whether one anomalous modality is enough or several modalities should jointly contribute.
- It is computationally feasible because the modality distances already exist.

Do **not** claim novelty yet. A targeted literature review is needed on multimodal anomaly-score aggregation, max/mean/softmax/log-sum-exp, late fusion, and multimodal bankruptcy prediction.

## 7. Recommended next experiment: direct aggregation-function ablation

This should be the next phase, instead of continuing to stack features onto `d_max`.

Use the same per-filing distances:
- `dF = d_financial`
- `dM = d_mda`
- `dA = d_auditor`

Compare pre-specified candidates:
1. **Max:** `max(dF, dM, dA)` — original MABAD aggregation.
2. **Mean:** arithmetic mean.
3. **Median:** median.
4. **Top-two mean:** mean of the two largest distances.
5. **Power mean:** a small pre-declared set such as p=2 and p=4; select only on validation.
6. **Softmax/log-sum-exp:** a small pre-declared temperature grid, selected only on validation.
7. **Max + range:** retain as an exploratory comparator, not an assumed winner.

Protocol:
1. Verify exact rows, labels, split, missing-value handling and score direction from source files/code.
2. Use training data for any learned transformations where appropriate; use validation for pre-specified model selection.
3. Freeze the chosen rule/hyperparameters before the final test comparison. Do not choose the winner using test AP and report that same test as confirmatory.
4. Report AP, ROC-AUC, Precision@K and Recall@K for K=10, 20, 50, 100, and baseline prevalence.
5. Use paired uncertainty estimates that preserve class structure; do not silently discard zero-positive bootstrap resamples.
6. Report validation folds, uncertainty, all candidates and selection details—not just the best result.
7. Be cautious with temporal slices containing very few positives.
8. Keep rows, labels, source distances and split identical across candidates.
9. If a logistic/calibration layer is fit on validation labels, explicitly call it a validation-fitted combination model, not a purely unsupervised MABAD aggregation.
10. Save scripts, configuration, predictions, tables and a report for reproducibility.

Fixed mathematical aggregators should be compared directly. Any tunable parameter must be selected using validation only. A learned combination model must have a clearly justified fitting protocol.

## 8. Decision criteria

Proceed toward a paper contribution only if:
- AP or practical top-K metrics improve meaningfully.
- The improvement is supported by validation/repeated temporal evaluation, not just one favorable test metric.
- The result is robust and not driven entirely by one modality or small subset.
- The protocol is leakage-safe and reproducible.
- A literature review establishes a clear distinction from prior work.

If alternatives do not robustly beat max, report that and revise or abandon the hypothesis. Do not force a novelty claim.

## 9. Instructions for Gemini

Treat this file as the project log and the supplied `research/` archive as the source of scripts, data products and reports. Before running experiments:

1. Inspect the archive tree and read Phase 0, Phase 0.5, audit, incremental-value and distance-geometry scripts/reports.
2. Match every reported metric to the exact script and CSV. Recompute ambiguous results; do not silently merge runs.
3. Check whether `distance_geometry_analysis.py` reconstructs `disagreement_std` as raw population standard deviation; that may differ from Phase 0's normalized disagreement.
4. Verify split definitions, labels, duplicates, row counts and prediction sources.
5. Do not retrain/reimplement MABAD unless the saved artifact chain is unusable.
6. Create a new isolated phase, e.g. `research/phase1_aggregation/`, and write the experiment protocol before examining test outcomes.
7. Keep test data out of model selection. Because earlier exploration used the test set, state that this is exploratory; if feasible, identify a later untouched holdout or propose another honest confirmation plan.
8. Return a concise recommendation—proceed, revise, or stop—with weaknesses and negative findings included.

## 10. Relevant paths

- `research/phase0/outputs/`
- `research/phase0/outputs/active_finbert_mabad_valtest_modality_distances.csv`
- `research/phase0/outputs/active_finbert_mabad_valtest_diagnostics.csv`
- `research/phase0/outputs/phase0_decision.md`
- `research/phase0_5/phase0_5_analysis.py`
- `research/phase0_5/phase0_5_report.md`
- `research/phase0_5/audit/phase0_5_audit.py`
- `research/phase0_5/audit/phase0_5_audit_report.md`
- `research/phase0_5/incremental_value_analysis.py`
- `research/phase0_5/distance_geometry_analysis.py`
- `research/phase0_5/distance_geometry/`
- Robustness package from this ChatGPT session: `distance_geometry_robustness/robustness_report.md`, `test_model_comparison.csv`, `validation_5fold_summary.csv`, `correlations.csv`, `yearly_test_comparison.csv`. This package may not yet be in the repository archive; preserve its provenance if copying it into the project.

## One-paragraph summary

We want a defensible research contribution to MABAD bankruptcy prediction without wasting time reimplementing a baseline whose saved artifacts are usable. Phase 0 audited the dataset and reconstructed modality-specific anomaly distances from the saved FinBERT MABAD artifact. The initial hypothesis that cross-modal disagreement signals unreliability was weakened by rare-positive instability, flawed bootstrap handling and confounding with overall anomaly magnitude. An exploratory comparison suggested that adding the spread/range of modality distances to the baseline maximum may improve test AP and top-K recall, but the evidence is not enough to claim an independent signal or a robust method. The next principled step is a pre-specified aggregation-function ablation (max, mean, median, top-two mean, power mean and softmax/log-sum-exp), with selection on validation and careful held-out evaluation. The potential novelty is a demonstrated, interpretable improvement to multimodal anomaly-distance aggregation—only if it survives rigorous evaluation and a literature review.
