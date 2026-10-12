# Track A static contradiction-score MVP

This directory contains a standalone prototype for an explicit, inspectable
financial-narrative contradiction feature. It follows the Track A formulas in
`research/idea (1).md` and does not modify MABAD, the Phase 4 scripts, or any
existing baseline artifact.

## Scope and status

This version is static only. No temporal contradiction score is implemented.
The repository has a date suffix in `uid`, but no verified filing/publication
date. The suffix is used only for the existing chronological split and for
provisional ordering metadata.

Narrative acknowledgement values are produced by a transparent rules extractor.
They are explicitly labelled `unvalidated_proxy`; they are not human labels,
ground truth, generic sentiment, or evidence of management intent.

The method is a proposed research prototype. It makes no claim of novelty,
validity, or predictive improvement.

## Files

- `topic_config.json`: configured topics, exact XBRL fields, ratio definitions,
  adverse direction, and weights.
- `financial_topic_scores.py`: training-only ratio construction and empirical
  adverse percentiles.
- `narrative_claims.py`: topic-anchor plus explicit-condition-cue extraction,
  sentence evidence, uncertainty flags, and separate auditor evidence.
- `contradiction_score.py`: static `B`, `A`, `S`, `R`, coverage, missing topics,
  and JSON output contract.
- `contradiction_evaluation.py`: data loading, existing split reproduction,
  feature-quality audit, and validation-only controlled comparisons.
- `test_track_a.py`: behavioral, boundedness, reproducibility, and future-row
  invariance tests.
- `annotation_template.csv`: human-review template for claim extraction.

## Formulas

For each configured ratio, the numerical value is calculated only when its
denominator is finite and strictly positive. Missing numerator, denominator,
invalid denominator, and infinite ratio all produce unavailable evidence; no
alternate XBRL field is silently substituted.

Each ratio is converted to an adverse-direction percentile using only the rows
passed to `FinancialTopicScorer.fit`. Topic severity `B` is the mean of the
available configured ratio severities. A topic is numerically available when at
least one configured ratio has a training reference and a valid current ratio.

For an applicable current-period claim:

```text
S = max(0, B - A)
R = max(0, A - B)
T = null  # temporal scoring is disabled in this version
topic_score = S
```

If there is no applicable claim, `A`, `S`, `R`, and `topic_score` are null. The
topic is not eligible. Silence is never converted into `A = 0`.

The overall score is the configured nonnegative-weighted mean of eligible topic
scores. Coverage is the eligible topic weight divided by total configured topic
weight. If no topic is eligible, the overall score is null and coverage is zero.
Coverage is not confidence.

## Run

From the repository root:

```bash
./.venv/bin/python -m unittest research.contradiction.test_track_a
./.venv/bin/python research/contradiction/contradiction_evaluation.py
```

Outputs are written only to `research/contradiction/outputs/`:

- `contradiction_features.csv`: one row per input UID, including the JSON
  topic payload and join-preserving metadata.
- `contradiction_evidence.csv`: management candidate sentences and separate
  auditor sentences.
- `financial_topic_components.csv`: raw ratios, training-only severity values,
  and availability columns.
- `feature_quality_summary.csv`: coverage, missingness, and positive counts by
  existing train/validation/test split.
- `controlled_validation_comparison.csv`: validation-only grouped comparisons
  using frozen held-out MABAD distances, when the Phase 4 input is available.
- `feature_schema.json`, `run_metadata.json`, `controlled_validation_metadata.json`,
  and `research_report.md`.

The script does not calculate test metrics. The existing later test period was
previously inspected and must not be described as an untouched holdout.

## Interpretation limits

The numerical ratios are sparse and XBRL concepts are heterogeneous. The
rules extractor requires a topic anchor and an explicit condition cue, but it
does not resolve all scope, discourse, negation, boilerplate, or mitigation
cases. Past-only and future/conditional claims are retained as evidence but
excluded from the current acknowledgement aggregate. Human annotation is the
next required step before treating `A` as a validated measurement.
