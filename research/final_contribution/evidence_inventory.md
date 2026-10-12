# Evidence Inventory and Reconciliation

## Scope and evidence standard

This inventory was reconstructed from the saved scripts, reports, CSV/JSON outputs, archived Phase 3 and Phase 4 output bundles, and the Track A audit. It separates a metric reproduced from a row-level artifact from a metric present only in a prior report. The repository test split was examined in earlier exploratory work and is not treated as an untouched confirmation set.

## Dataset and artifact chain

The active FinBERT MABAD artifact chain contains 39,041 synchronized firm-filings, 192 positive filings, and 4,855 companies. The documented chronological split is 29,752 train rows/126 positives through 2017, 2,908 validation rows/19 positives in 2018, and 6,381 test rows/47 positives in 2019–2021. The Phase 0 and Phase 4 distance files carry the three frozen modality distances: financial, MD&A, and auditor text.

The active artifacts are usable for post-hoc distance aggregation. They do not establish that every historical MABAD result was reproduced end-to-end because original final per-row predictions were not saved and the full baseline recreation previously encountered a missing imputer artifact.

## Phase-by-phase evidence

| Phase | Research question / method | Evidence and metric reconciliation | Status and limitations |
|---|---|---|---|
| 0 / 0A | Can saved MABAD artifacts reconstruct auditable modality distances and does completeness carry ranking signal? | `phase0c_diagnostics.json` reports combined validation+test `d_max` AP 0.098907, ROC-AUC 0.900348, R@50 0.136364, R@100 0.242424. Its validation-only `d_max` AP 0.241456 and ROC-AUC 0.947860 match Phase 4 exactly. `phase0a_summary.json` reports 39,041 rows/192 positives and weak missingness-only ranking (best AP about 0.0053). | Diagnostic reconstruction, not an independent reproduction of the original paper. The combined validation+test metrics are descriptive and not method-selection evidence. |
| 0.5 | Does cross-modal disagreement signal unreliable MABAD ranking? | Initial report compared tiny disagreement quantiles, including validation Q1 with one positive and Q2 with two positives. Audit confirms quantile boundaries were fit on validation but rejects subgroup AP and bootstrap inference because zero-positive resamples were discarded. | Original “strong evidence” conclusion is invalid. Disagreement is correlated with `d_max`; independent reliability value is unestablished. |
| Incremental value / distance geometry | Does disagreement or range add to `d_max` in a validation-fitted logistic model? | Reports show exploratory test AP improvements for some range models, e.g. distance-geometry G1 AP 0.1138 versus 0.0833 for `d_max`, but the same test split had already been explored. The reports contain inconsistent range estimates across runs. | Test-informed exploratory leads only. Not used to select the final direction. |
| 1 | Does a direct, fixed aggregation rule outperform raw max? | `aggregation_metrics.csv` / report: validation AP for raw max 0.241456; raw p=4 0.311121; validation ROC-AUC 0.947860 versus 0.940846. Test results are explicitly exploratory. Five-fold validation is highly variable. | Exploratory candidate ablation with 19 validation positives. The generalized-mean family and `p=4` were among multiple examined candidates. |
| 2 | Can scale-aware/reliability-weighted fusion improve aggregation? | Preliminary five-fold mean APs: raw mean 0.3425, damped financial percentile 0.3313, IQR-weighted 0.3293, raw max 0.2878. The source reports very large fold SDs (0.27–0.39). | Preliminary only; no stable selection basis. Low financial IQR motivates scale diagnostics but does not establish low modality reliability. |
| 3 | Can leakage-aware grouped OOF aggregation reproduce a candidate ranking? | Archived `validation_oof_summary.csv`: raw p=4 AP 0.311121, ROC-AUC 0.940846; raw max AP 0.241456, ROC-AUC 0.947860. The archived 500-draw p=4-minus-max AP interval is [-0.033183, 0.202783]. | Same 2018 validation cohort and a candidate set, not independent confirmation. Test values in the archive are exploratory. |
| 4 | Do fixed raw, percentile, and robust aggregations hold under a pre-specified grouped OOF protocol? | Row-level OOF file has 2,908 rows, 19 positives, 2,906 companies, no duplicate UIDs, five `StratifiedGroupKFold` folds. Raw p=4 AP 0.311121; raw max AP 0.241456; AP difference +0.069664. The saved 2,000-draw company-cluster bootstrap interval is [-0.041146, 0.189565]. | Best available controlled evidence. The interval includes zero; selection occurred within exploratory validation comparisons; no clean independent holdout exists. |
| Track A contradiction | Does explicit numerical–narrative under-acknowledgement add value beyond MABAD? | Feature coverage exists for 2,696/2,908 validation rows. The reported augmented APs are 0.174541 for gap+coverage, 0.235648 for severity without gap, and 0.207077 for coverage-only versus frozen `d_max` AP 0.241456. | Unsuccessful preliminary predictive result. The comparison is unfair because the baseline is direct `d_max` while alternatives are new fold-trained logistic models with a different fold seed; augmented per-row OOF predictions were not saved. Narrative claims are unvalidated rules proxies. |

## Phase 4 reconciliation performed in this workspace

`cached_oof_reproduction.py` recomputed raw maximum and raw p=4 scores from the three frozen distances in the Phase 4 validation OOF file. It verified:

- 2,908 unique UIDs, 19 positives, and 2,906 companies;
- exact match of recomputed scores to the saved OOF score columns within `1e-12` absolute and relative tolerance;
- the same labels and saved company-grouped folds for baseline and proposal;
- pooled AP 0.241456 for max and 0.311121 for p=4;
- pooled ROC-AUC 0.947860 for max and 0.940846 for p=4; and
- exact reproduction of the saved 2,000-draw company-cluster AP interval [-0.041146, 0.189565].

The reproduction loaded no test file or test labels. It confirms calculation and provenance; it does not create new independent predictive evidence.

## Candidate-direction assessment

| Candidate | Evidence available | Feasibility now | Defensible conclusion |
|---|---|---|---|
| A. Fixed, topic-agnostic distance aggregation | Direct Phase 1/3/4 comparison, row-level Phase 4 OOF predictions, exact reproduction | High | Strongest available direction, but only a preliminary MABAD-specific empirical lead. |
| B. Reliability-aware fusion | Scale mismatch diagnostics and unstable preliminary weights; disagreement is confounded with anomaly magnitude | Moderate implementation, weak evidence | Not ready. A new mechanism needs a prospectively defined reliability target and independent validation. |
| C. Explicit contradiction score | Rich evidence export, but semantic extraction failures, no verified human labels, and unfavorable preliminary metrics | Low for valid predictive claim | Stop predictive development until independent annotation validates claim applicability and severity. |
| D. Interpretable diagnostic framework | Evidence tables and coverage are usable, but the acknowledgement proxy remains unvalidated | Moderate, non-predictive | Potentially useful as an engineering audit tool, not yet a research contribution. |
| E. Test-informed distance geometry models | Exploratory test gains exist but protocol has already seen test outcomes | Low for confirmation | Not suitable as the selected contribution without new data. |

## Selected direction

Select A: **a reproducible ablation of hard-maximum versus fixed generalized-mean aggregation of frozen MABAD modality distances**. It directly targets a baseline mechanism, uses identical rows and group-aware folds, has a compact interpretable equation, and has the most complete saved artifact chain. The selected result remains statistically uncertain and cannot be called a validated improvement.

## Evidence gaps that block a publishable superiority claim

1. A genuinely untouched later or external evaluation cohort, with a frozen method and analysis protocol.
2. More positive events for stable AP uncertainty estimation.
3. A pre-registration that fixes `p=4` before evaluating the confirmation cohort.
4. For Track A, verified dual-human narrative annotations and a fair common-scoring-protocol comparison.
