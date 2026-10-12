# Human Review Guide: Narrative Claim Sample

This file accompanies `narrative_claim_review_ready.csv`, a 30-row sample of rules-based extracted narrative claims from the Track A contradiction MVP. The machine-generated fields are retained for audit, but they are not human labels or validated ground truth. Do not treat this sample as already reviewed.

## Review procedure

Review the complete extracted sentence first. Where available, look up the surrounding MD&A context using `filing_uid` and `sentence_id`. Do not judge from a keyword alone. The assigned topic and predicted acknowledgement severity are prompts for review, not answers. Do not use the bankruptcy label to decide whether a claim is applicable or severe.

Leave a field blank only if the review has not yet been performed. Use `uncertain` or `unclear` when the available sentence and context do not support a reliable decision. Put the reason for an uncertain decision in `human_notes`.

## Annotation fields

### `human_topic_relevance`

- `relevant`: the sentence genuinely concerns the assigned financial topic.
- `not_relevant`: the sentence does not concern that topic, even if it contains a related word.
- `uncertain`: the topic cannot be judged reliably without unavailable context.

Topic relevance asks whether the sentence is about the topic itself, not whether it merely mentions a term associated with it. For example, a sentence about a derivative used to hedge interest-rate exposure is not automatically a leverage or liquidity claim.

### `human_claim_applicable`

- `yes`: the sentence makes a meaningful claim about the company’s financial condition, exposure, obligation, or ability to meet the relevant condition.
- `no`: it is only a definition, accounting-policy description, legal provision, transaction description, historical aside, or other term mention without a meaningful condition claim.
- `uncertain`: applicability cannot be determined from the available evidence.

A topic-relevant sentence can still be inapplicable. A loan agreement may describe an event-of-default clause without saying that the company is in default. A derivative note may discuss notional amounts without describing adverse leverage or liquidity.

### `human_acknowledgement_severity`

Use the following ordinal scale for acknowledgement of an adverse condition in the assigned topic:

- `0`: no acknowledgement of an adverse condition.
- `1`: vague or indirect acknowledgement; the condition is hinted at but not clearly described.
- `2`: clear acknowledgement of the adverse condition.
- `3`: explicit, specific acknowledgement, such as a named deterioration, breach, inability to pay, material shortfall, or quantified adverse exposure.
- `uncertain`: the sentence cannot be judged reliably.

This field measures acknowledgement, not overall sentiment. A positive statement that explicitly denies a problem may be relevant to `human_negation`, but it normally does not count as acknowledgement of the adverse condition unless the sentence also clearly acknowledges the underlying issue.

### `human_negation`

- `present`: the sentence negates, denies, or explicitly rules out the relevant adverse condition.
- `absent`: no such negation is present.
- `uncertain`: the wording or context is ambiguous.

Consider scope. “We do not expect a liquidity shortfall” negates an expected shortfall, while a negation inside a conditional or quoted legal clause may not describe the company’s current condition.

### `human_mitigation`

- `present`: the sentence provides a mitigating explanation, counterevidence, protective factor, or qualification relevant to the adverse condition.
- `absent`: no mitigating explanation or qualification is provided.
- `uncertain`: the role of the explanation cannot be determined.

Examples include available committed facilities, collateral, refinancing plans, waivers, or a qualification that an exposure is remote or hedged. A generic positive adjective is not mitigation by itself.

### `human_timing`

- `current`: concerns the company’s current or reporting-period condition.
- `historical`: concerns a past condition or completed event without presenting it as current.
- `future`: concerns an expected, planned, or conditional future condition.
- `unclear`: timing cannot be established from the sentence and available context.

Use the reporting period and surrounding context where needed. A sentence containing a past-tense event may still explain a current exposure; record the timing that best describes the financial claim being reviewed.

### `human_notes`

Provide a short explanation for borderline decisions, context used, misleading keywords, or why a sentence was marked uncertain. Do not write a label in this column unless the structured field has also been reviewed.

## Common misleading matches

The following examples do not establish financial distress by themselves:

- Derivatives terminology such as “credit default swap,” “interest-rate swap,” “notional amount,” or “cash flow hedge.” These may describe hedging or fair-value accounting rather than leverage or liquidity stress.
- Boilerplate accounting language about impairment, losses, cash flows, debt, or financing that describes a policy, disclosure requirement, or measurement method rather than the company’s adverse condition.
- Loan-covenant and event-of-default references that state what would happen if a covenant were breached. A conditional legal clause is not evidence that a breach occurred.
- Historical comparisons, acquired-company information, or completed refinancing language that is not describing the current reporting-period condition.
- Generic reassurance such as “adequate,” “strong,” or “stable” without a specific claim about the assigned topic. This is not automatically a clear acknowledgement or mitigation.
- Sentences containing a topic word in a list, table description, contract summary, or definition without a claim about the company’s condition.

Review the full sentence and its context rather than counting keywords. If the context needed to distinguish a current financial claim from boilerplate, a hypothetical, a historical statement, or a legal/derivatives disclosure is unavailable, use `uncertain`/`unclear` and explain the limitation.

## Controlled vocabulary

The intended values are:

- `human_topic_relevance`: `relevant`, `not_relevant`, `uncertain`
- `human_claim_applicable`: `yes`, `no`, `uncertain`
- `human_acknowledgement_severity`: `0`, `1`, `2`, `3`, `uncertain`
- `human_negation`: `present`, `absent`, `uncertain`
- `human_mitigation`: `present`, `absent`, `uncertain`
- `human_timing`: `current`, `historical`, `future`, `unclear`
- `human_notes`: free text

Human review has not been completed by preparing this file. The rules-based extractor remains an unvalidated proxy until the completed annotations are independently checked and analyzed.
