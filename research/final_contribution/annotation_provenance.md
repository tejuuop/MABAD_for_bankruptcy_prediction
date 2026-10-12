# Narrative Review-File Provenance Audit

## Files inspected

- `research/contradiction/outputs/narrative_claim_review_sample.csv`
- `research/contradiction/outputs/narrative_claim_review_ready.csv`
- `research/contradiction/outputs/human_review_file_summary.md`
- `research/contradiction/outputs/HUMAN_REVIEW_GUIDE.md`
- `research/contradiction/outputs/audit_report.md`

## Observed state

The original sample has 30 rows. Its legacy `human_judgement` and `human_notes` fields are blank in all 30 rows. The review-ready copy has the same 30 rows and preserves the original identifiers and machine-extraction fields. It contains populated values in all 30 rows for:

- `human_topic_relevance`;
- `human_claim_applicable`;
- `human_acknowledgement_severity`;
- `human_negation`;
- `human_mitigation`;
- `human_timing`; and
- `human_notes`.

`human_judgement` remains blank in the review-ready file. The review-ready file was modified after the original sample. The nearby summary still describes the fields as blank and says that no automatic annotation was added, so it is stale relative to the populated review-ready file.

## Provenance finding

The repository provides no signed annotation log, reviewer identity, timestamped review protocol, adjudication record, or Git history for these untracked files that establishes a human annotator. The populated fields therefore cannot be treated as human-verified labels.

For all downstream research purposes, classify them as **unverified provisional annotations of unknown/AI-assisted provenance**. They are not gold labels, must not be used to train or calibrate a claim extractor, and must not support an annotation-accuracy claim.

## Label-status matrix

| Artifact or field | Provenance status | Permitted use now | Prohibited use now |
|---|---|---|---|
| Extractor fields (`acknowledgement_severity`, `claim_applicable`, `negated`, `mitigating_explanation`, `uncertainty`) | Known rules-based machine proxy | Error analysis and audit | Ground-truth narrative measurement |
| Original sample’s blank `human_judgement` and `human_notes` | No labels present | Review template | Any accuracy estimate |
| Populated structured `human_*` fields in review-ready copy | Unverified provisional / unknown human provenance | Illustrative review queue only | Training, evaluation, gold-label claims, or extractor calibration |
| Independently completed, documented human annotation with adjudication | Not available | None until created | N/A |

## Consequence for Track A

The Track A contradiction score remains a rules-based prototype with unvalidated narrative acknowledgement. Its negative, unfairly scored predictive comparison is not used in the selected aggregation experiment. The highest-value prerequisite for reviving Track A is a pre-registered dual-human annotation exercise with a documented codebook, sentence context, disagreement resolution, and held-out annotation set.
