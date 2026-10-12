# Human Review File Summary

## Files

- Original sample inspected: `research/contradiction/outputs/narrative_claim_review_sample.csv`
- Review-ready copy: `research/contradiction/outputs/narrative_claim_review_ready.csv`
- The original sample was not modified.

## Row and column summary

- Rows in original sample: 30
- Rows in review-ready copy: 30
- Original source columns preserved exactly in the review-ready copy:
  - `filing_uid`
  - `sentence_id`
  - `topic`
  - `sentence`
  - `acknowledgement_severity` (machine-predicted proxy)
  - `human_judgement` (legacy blank field retained from the source sample)
  - `human_notes`
  - `claim_applicable`
  - `negated`
  - `mitigating_explanation`
  - `uncertainty`
  - `extraction_method`
  - `narrative_status`

Required annotation columns are present and blank:

- `human_topic_relevance`
- `human_claim_applicable`
- `human_acknowledgement_severity`
- `human_negation`
- `human_mitigation`
- `human_timing`
- `human_notes` (already existed in the original sample and was preserved)

## Validation results

- Exact 30-row check: passed.
- Original source fields preserved row-for-row: passed.
- No duplicate full rows introduced: passed.
- All required human annotation fields blank: passed.
- Existing `human_judgement` and `human_notes` fields in the original sample were blank: passed.

## Context limitations

- The review CSV contains the extracted sentence and identifiers, not the full MD&A filing context. Reviewers should retrieve surrounding text by `filing_uid`/`sentence_id` where possible.
- The sample does not provide a verified filing/publication date; any UID date should not be treated as a verified disclosure date.
- The machine acknowledgement severity, `claim_applicable`, `negated`, `mitigating_explanation`, and `uncertainty` fields are rules-based proxy outputs, not human labels.
- The file does not include independent human labels, a validated narrative taxonomy, or a separate auditor-text review.
- Context may be needed to distinguish current conditions from historical, future, conditional, legal, derivatives, accounting-policy, or loan-covenant language.

No model experiment was rerun and no automatic human annotation was added.
