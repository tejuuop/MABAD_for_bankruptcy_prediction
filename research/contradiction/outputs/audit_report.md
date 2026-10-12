# Audit report — Track A contradiction MVP

Audit scope: read-only review of `README.md`, `research_report.md`, the Track A
feature/evaluation code, generated CSVs, and the existing Phase 4 metadata. No
model or baseline files were changed. The 30-row review sample created during
this audit is [narrative_claim_review_sample.csv](narrative_claim_review_sample.csv).
Its `human_judgement` and `human_notes` columns are intentionally blank.

## Executive conclusion

The static feature construction is mostly faithful to the stated Track A
formula, and the generated joins are internally consistent. The controlled
comparison is not a fully fair baseline-versus-augmentation experiment:

1. the frozen MABAD baseline is evaluated directly as `d_max`;
2. the augmented models are new fold-trained logistic models using `d_max` plus
   Track A features;
3. the augmented models use validation-fold preprocessing and a different fold
   seed from the saved Phase 4 OOF baseline; and
4. augmented per-row OOF predictions were not saved, preventing a paired
   company-cluster uncertainty calculation.

The narrative extractor also produces many obvious topic and severity false
positives in the stratified sample. It should not be treated as a validated
acknowledgement measurement.

## 1. Evaluation validity

### Rows, labels, and groups

The comparison starts from the existing Phase 4 distance file and selects its
`split == "val"` rows in `contradiction_evaluation.py:174-180`. The generated
metadata reports 2,908 validation rows and 19 positives. The feature join is a
one-to-one UID merge (`contradiction_evaluation.py:180-183`). An independent
audit check found:

- 2,908 Phase 4 validation rows;
- zero missing contradiction-feature joins;
- zero label mismatches between the feature output and Phase 4 validation file;
- zero duplicate UIDs in the validation prediction file;
- 2,906 validation companies and four rows belonging to repeated companies.

Thus, the rows and labels are aligned.

### Folds

The augmented models use `StratifiedGroupKFold(n_splits=5, shuffle=True,
random_state=20261011)` (`contradiction_evaluation.py:38-39, 188-188`) and pass
`company_id` as the grouping vector (`contradiction_evaluation.py:186-188,
206-213`). Repeated company rows are therefore kept together for the augmented
OOF predictions.

However, the saved Phase 4 baseline OOF folds were produced with seed `20261012`
according to `phase4_run_metadata.json`. Reconstructing the splits showed:

- seed `20261012`: 2,908/2,908 assignments match the saved Phase 4 `oof_fold`;
- seed `20261011`: only 610/2,908 assignments match those saved folds.

More importantly, the baseline in the new comparison is not assigned to folds at
all. It is scored directly from `d_max` at
`contradiction_evaluation.py:190-191`. Therefore the baseline and augmented
models were not evaluated on exactly the same fold assignments.

### Fitted transformations and tuning

The augmented pipeline is:

```text
SimpleImputer(strategy="median", add_indicator=True)
StandardScaler()
LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20261011)
```

This is defined at `contradiction_evaluation.py:157-162` and fitted separately
on each validation training fold at `contradiction_evaluation.py:206-212`.
Consequently, fold-training medians, missingness indicators, scaling parameters,
and logistic coefficients are learned from validation rows. This is acceptable
for an OOF CV estimator, but it means the evaluation is not a frozen-feature
comparison. There is no threshold selection, feature selection, or hyperparameter
search in the code. The feature blocks are hard-coded before evaluation.

Numerical Track A percentile references are fitted on the chronological training
split only (`contradiction_evaluation.py:229-246`), not on validation or test.

## 2. Feature integration and fairness

The comparison merges Track A features into frozen Phase 4 MABAD distances by
UID (`contradiction_evaluation.py:179-185`). The baseline is raw frozen `d_max`.
Each augmented model receives `d_max` plus one of these predeclared feature
blocks (`contradiction_evaluation.py:193-202`):

- contradiction score, coverage, and eligible-topic count;
- numerical and narrative severities plus availability indicators, without the
  gap;
- coverage and eligible-topic count only.

The code therefore trains a new second-stage logistic model. It does not merely
learn a coefficient for contradiction while leaving the baseline scoring rule
unchanged. MABAD itself is not retrained; the comparison uses frozen held-out
MABAD distances, as recorded in `controlled_validation_metadata.json`.

This is reproducible from the source code and fixed seed, but it is not fully
fair as reported because the baseline is a direct rank score while the
augmented variants are fold-trained probability models. A fair comparison would
put the baseline through the same fold protocol/model family or compare all
feature blocks with a predeclared common scorer.

The generated CSV stores aggregate fold metrics only. The augmented OOF arrays
remain local to `controlled_validation_comparison()` and are not written. This
prevents independent row-level reproduction and paired uncertainty estimation.

## 3. Score correctness

### Numerical ratios

The configured ratios are explicit in `topic_config.json`:

- liquidity: cash/assets, current ratio, operating cash flow/assets;
- leverage: liabilities/assets, long-term debt/assets, interest expense/revenue;
- profitability: net income/assets, operating income/assets, net income/revenue.

The exact fields and adverse directions are in `topic_config.json:9-98`.
Ratio construction converts fields to numeric and requires a finite denominator
strictly greater than zero (`financial_topic_scores.py:81-86`). Missing fields,
missing numerators, invalid denominators, and infinite ratios become unavailable.
No alternate XBRL concept is silently substituted.

### Training-only percentiles

`FinancialTopicScorer.fit()` computes each ratio reference distribution only from
the passed training frame (`financial_topic_scores.py:88-101`).
`_adverse_percentile()` uses the training reference with `searchsorted`, reverses
the percentile for lower-is-worse ratios, and clips to `[0,1]`
(`financial_topic_scores.py:103-115`). The current evaluation calls `fit()` on
the existing training split before transforming chunks (`contradiction_evaluation.py:229-246`).

### Missingness and aggregation

Within a topic, the implementation averages the available configured ratio
severities (`financial_topic_scores.py:140-151`). All three configured topic
weights are 1.0 (`topic_config.json:9-72`), so the overall eligible-topic score
is an equal-weight mean. Topic eligibility requires both numerical severity and
an applicable narrative claim (`contradiction_score.py:53-70`). The score is
then aggregated over eligible topics only, with coverage equal to eligible
weight divided by total configured weight (`contradiction_score.py:103-126`).

If no applicable narrative claim exists, `A` is null and the topic is ineligible;
silence is not converted to zero acknowledgement. This is correctly reflected
in `aggregate_current_claims()` (`narrative_claims.py:215-232`) and the generated
feature outputs.

### Static formula boundary

The implementation correctly computes `S = max(0, B-A)` and
`R = max(0, A-B)` (`contradiction_score.py:56-65`). Temporal `T` is explicitly
null and disabled (`contradiction_score.py:64-77`, `topic_config.json:4-7`).

One documentation limitation remains: equal ratio weights are implemented by a
plain mean but are not represented as explicit per-ratio weights in the config.
The behavior is deterministic, but the weighting choice should be made explicit
before a confirmatory experiment.

## 4. Narrative extraction sample

The audit sample contains 30 extracted management claims: three topics, with
three severity-0, four severity-0.55, and three severity-1.0 claims per topic.
It contains the UID, sentence ID, sentence, predicted severity, applicability,
negation flag, mitigation flag, uncertainty, and blank human-review fields:

[narrative_claim_review_sample.csv](narrative_claim_review_sample.csv)

This is a deterministic sample of extractor output, not a human-reviewed
annotation set. No human labels were fabricated.

## 5. Evidence and narrative quality

The extractor requires a topic keyword and an explicit cue
(`narrative_claims.py:93-109`), but the audit sample shows substantial semantic
noise:

- A leverage claim was extracted from a sentence about “adequate liquidity” and
  scheduled debt payments. It is not a clean leverage-condition claim.
- A leverage claim with severity 0.55 came from a sentence about a debt security
  being written down, which is not necessarily company leverage pressure.
- A liquidity claim with severity 0.55 came from a derivative cash-flow-hedge
  accounting sentence rather than liquidity condition.
- A profitability severity-1 claim came from “credit default” derivatives;
  the `default` substring activates the critical rule even though the sentence
  is about a financial instrument, not corporate default.
- Legal covenant and event-of-default boilerplate is frequently treated as
  severe leverage or liquidity acknowledgement.

The severity order is critical, then moderate, then reassurance
(`narrative_claims.py:102-109`). Negation is only recorded as a flag at
`narrative_claims.py:149-150`; it does not alter severity or exclude the claim
from aggregation. For example, a sentence containing “liquidity” and “material
adverse” can receive severity 1.0 even when it says the effect is *not*
expected. This is a confirmed extraction/scoring defect.

Mitigating explanations are detected and recorded in uncertainty text
(`narrative_claims.py:151-154`), but they are still included in the maximum-
severity acknowledgement aggregate (`narrative_claims.py:220-232`). They do
not reduce the gap or make the topic ineligible. This is another confirmed
implementation limitation relative to the intended mitigation behavior.

Past/future context is partially handled by the narrow regexes at
`narrative_claims.py:41-42,112-121`; only claims marked current are aggregated.
The sample includes conditional language such as “if the purchase option is not
exercised” that is still marked current, demonstrating incomplete context
handling.

The evidence volume is large: 1,025,805 management candidates and 96,323
separate auditor evidence rows. The volume is descriptive evidence retention,
not evidence that the claims are semantically valid.

## 6. Leakage, joins, and temporal safety

The generated feature output has 39,041 rows and 39,041 unique UIDs. The
streaming join checks duplicate identifiers, missing IDs, and exact UID-set
equality (`contradiction_evaluation.py:58-73`). The final writer also checks row
count, UID uniqueness, label alignment, and score bounds
(`contradiction_evaluation.py:255-266`). The audit found zero Phase 4 validation
join failures and zero label mismatches.

The scorer is row-local after fitting: it does not use company history, later
observations, or temporal features. The automated future-row test verifies that
adding a later row after fitting does not change an earlier score
(`test_track_a.py:101-110`). This establishes invariance for the tested API
behavior, not provenance of the underlying filing date.

The main temporal limitation remains: the code derives `prediction_cutoff` from
the UID suffix and explicitly labels it
`uid_date_order_only_not_verified_publication_date`
(`contradiction_score.py:16-25`). No filing/publication date is available, so
the implementation cannot establish that all inputs were public by the stated
cutoff.

The split function also uses positive labels from the entire loaded dataset to
choose the train/validation boundaries (`contradiction_evaluation.py:76-92`).
That reproduces the repository's existing split rule, but it is not a fully
prospective partitioning procedure because later outcomes help define the
boundaries.

The Phase 4 metadata records 2,707 companies shared between validation and test.
This is not by itself feature leakage because the score is row-local and test
metrics were not calculated, but it means the available data do not provide a
clean company-disjoint future holdout.

## 7. Metric uncertainty

All reported comparison rows contain 2,908 validation observations and 19
positives. The frozen direct baseline and grouped-OOF augmented point estimates
are:

| Comparison | AP | ROC-AUC | AP difference vs frozen baseline |
|---|---:|---:|---:|
| Frozen `d_max` baseline | 0.241456 | 0.947860 | — |
| MABAD + contradiction + coverage | 0.174541 | 0.942905 | -0.066915 |
| MABAD + severity without gap | 0.235648 | 0.936000 | -0.005808 |
| MABAD + coverage/missingness only | 0.207077 | 0.905321 | -0.034379 |

Exact paired company-cluster resampling for these contradiction comparisons
cannot be performed from the saved outputs because the augmented per-row OOF
predictions were not exported. Only aggregate fold metrics are present in
`controlled_validation_comparison.csv`.

The existing Phase 4 OOF artifact does support its own paired company-cluster
comparison: raw power mean p=4 minus original maximum had AP difference
`+0.069664`, with reported 95% interval `[-0.041146, +0.189565]`, based on 2,908
rows and 19 positives. That interval is contextual Phase 4 evidence, not an
uncertainty interval for the contradiction feature.

No test-set metrics were calculated in this audit.

## Final classification

### Confirmed implementation defects

1. The reported baseline and augmented comparisons do not share exactly the
   same fold assignments or scoring protocol. The baseline is direct `d_max`;
   augmented scores are fold-trained logistic predictions. The saved Phase 4
   folds use seed 20261012, while the augmented code uses 20261011.
2. Augmented OOF predictions are not saved, preventing exact paired bootstrap
   uncertainty and independent row-level reproduction.
3. Negation is recorded but not used to correct severity or eligibility.
4. Mitigating explanations are flagged but do not reduce or invalidate the
   contradiction score.
5. Keyword and cue rules produce clear semantic false positives in the audit
   sample, including legal boilerplate, accounting descriptions, and “credit
   default” instrument language.

### Methodological limitations

- Only 19 validation positives are available.
- UID dates are not verified filing/publication dates.
- Split boundaries are selected using positive labels across the full dataset.
- No untouched future or independent holdout is available; the later test period
  was previously examined.
- XBRL concepts are sparse and heterogeneous; the ratios are modeling proxies,
  not financial ground truth.
- The second-stage model is learned on validation folds, so the comparison is
  exploratory rather than a clean confirmatory estimate.

### Issues requiring human annotation

The review sample needs labels for topic relevance, current-period applicability,
acknowledged severity, negation scope, conditional/future/past timing, mitigating
explanations, and boilerplate/legal/accounting context. Annotation should be
performed independently of bankruptcy labels and used to calibrate or reject
the rules before another predictive experiment.

### Findings that remain inconclusive

- Whether an explicit contradiction feature adds information beyond MABAD.
- Whether the negative AP differences persist under a fair common scoring
  protocol and paired company-level uncertainty.
- Whether the rules-based narrative proxy can be calibrated into a valid `A`
  measurement.
- Any claim of novelty or research contribution.

The next valid step is human annotation and verified filing-date provenance,
followed by a predeclared, common-protocol OOF comparison that saves every
baseline and augmented prediction before any future-period evaluation.
