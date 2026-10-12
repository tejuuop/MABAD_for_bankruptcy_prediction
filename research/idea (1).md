# Contradiction Score for Our Bankruptcy Prediction Project

**Team handoff / proposed implementation specification**  
**Status:** Design proposal. No performance improvement or novelty has been established.  
**Target:** Extend the team's reproduced MABAD baseline with an explicit, explainable feature.

## 1. The idea in plain language

We want to measure whether management's description of a company's financial condition agrees with the financial evidence available at the same time.

Example:

- Financial evidence: cash is falling, debt is rising, and losses are increasing.
- Management statement: “Our liquidity remains strong.”
- Question: does that statement have supporting financial evidence or an explanation, such as a new credit facility?

A contradiction score should flag a supported mismatch for review. It should not flag every positive sentence in a weak company.

**Distress and contradiction are separate:** a financially weak company that accurately acknowledges its problems can have high bankruptcy risk and low contradiction. A low contradiction score must never override strong financial distress evidence.

The research hypothesis is:

> An explicit, topic-specific measure of disagreement, including disagreement in changes over time, may add predictive or explanatory value beyond MABAD's existing multimodal alignment.

## 2. What this adds to MABAD

MABAD already models multiple modalities and their alignment. It can potentially learn cross-modal inconsistency implicitly. We should not claim that MABAD ignores disagreement.

Our proposed extension makes disagreement an explicitly computed variable with identifiable financial inputs, source sentences, topic contributions, and time comparisons. We then test whether it adds value.

| Existing baseline | Proposed extension |
| --- | --- |
| Financial and narrative representations with multimodal anomaly detection | Additional topic-specific mismatch features |
| Cross-modal relationships learned by the model | Explicit rules or models for financial severity versus narrative acknowledgement |
| Baseline anomaly/risk output | Baseline output plus inspectable disagreement evidence |

Merely renaming an embedding distance “contradiction” would be a weak contribution. The value must come from the defined mechanism, explanations, and evaluation.

## 3. Two implementation tracks

### Track A: practical first version (recommended)

Start with matched financial topics and narrative claims. Compute static disagreement and, where historical observations exist, temporal disagreement. Export the component features and evidence for integration with MABAD.

This is a **new proposed MVP specification**, not an implementation of every mathematical operator in the earlier presentation/design note.

### Track B: experimental geometric version

The earlier NNCE design note proposes financial potentials, gradients and Hessians, narrative manifolds, geodesic divergence, curvature mismatch, Lie brackets, and adaptive reliability filtering.

These terms require fitted mappings, compatible geometry, stable derivatives, and substantial validation. A large formula alone does not establish semantic contradiction. Do not implement all terms merely to match the presentation. Start with Track A as an interpretable comparator, then add Track B terms individually if the data and experiments justify them.

If the project adopts Track A as its main algorithm, update the presentation to reflect that choice.

## 4. Input data and alignment

Inspect the actual repository and CSV schema before modifying code. Do not assume particular column names or that quarterly histories exist.

Required logical fields:

| Field | Purpose |
| --- | --- |
| Stable company identifier | Join modalities and build histories |
| Financial period end | Identify the accounting period |
| Filing/publication date | Determine when information became available |
| Financial ratios or underlying values | Measure numerical condition |
| MD&A text | Extract management claims |
| Auditor text, if available | Separate supporting or conflicting evidence |
| Next-year bankruptcy label | Evaluate the prediction task |

Use the dataset's actual reporting frequency. Annual data supports annual comparisons. Never manufacture quarterly narrative observations by interpolation and then treat them as real disclosures.

At prediction time t, use only filings and financial observations available by the prediction cutoff. A later bankruptcy label is an outcome, never an input to score construction.

## 5. Compare the same topic, period, and scope

Candidate topics:

| Topic | Candidate numerical evidence | Relevant narrative evidence |
| --- | --- | --- |
| Liquidity | Cash/assets, working capital, current ratio, operating cash flow | Liquidity adequacy, cash constraints, available funding |
| Leverage / debt pressure | Debt/assets, interest coverage, debt maturities when available | Debt burden, refinancing, covenant pressure |
| Profitability | ROA, operating margin, sustained losses | Profitability, losses, operating performance |
| Growth / operations | Revenue growth and available operating measures | Current demand, revenue decline, business performance |

Use only features that actually exist. Specify each ratio's definition, denominator safeguards, and distress direction. Validate topic mappings with finance knowledge.

Do not compare a future growth forecast with a current-year loss and automatically call it contradictory. Do not treat “no liquidity problem” as equivalent to “liquidity problem.” Consider explanations, negation, scope, and conditional language.

## 6. Numerical severity B[j,t]

For topic j at time t, derive a bounded numerical severity B[j,t] in [0,1]. Higher values mean greater numerical weakness for that topic.

A reproducible starting option:

1. Fit outlier limits and preprocessing on training observations only.
2. Convert each ratio into an adverse-direction training percentile. For example, lower cash/assets is more adverse, while higher debt/assets is more adverse.
3. Combine the available ratios within a topic using documented nonnegative weights that sum to one.
4. Export ratio contributions and missingness indicators.

Example interpretation: B[liquidity,t] = 0.85 means the selected liquidity features are relatively weak against the training reference. It does **not** mean an 85% bankruptcy probability.

Sector differences and ratio definitions can materially affect this measure. Percentile severity is a modeling choice, not financial ground truth. If a learned numerical model replaces it, fit that model using training data only and preserve a clear topic interpretation.

## 7. Narrative acknowledgement A[j,t]

For the same topic, extract applicable current-period management claims and estimate acknowledged severity A[j,t] in [0,1]. Higher values mean management acknowledges more severe weakness.

Suggested development process:

1. Retrieve sentences relevant to the financial topic, retaining nearby context.
2. Identify whether they concern the current condition, a past condition, or a future expectation.
3. Label a reviewed sample for topic, claim applicability, severity, negation, and mitigating explanations.
4. Build a transparent rules baseline or train a sentence classifier against these labels.
5. Aggregate applicable sentences using a documented policy, retaining conflicting claims and uncertainty.

A generic positive/negative sentiment score is insufficient. An encoder such as FinBERT supplies a representation or starting model; its output does not automatically equal acknowledgement of financial distress.

**Missing mention is missing evidence.** If no applicable claim exists, A is unavailable. Do not set A = 0 and call silence an optimistic claim. An omission feature can be researched separately with its own name and evaluation.

The numerical and narrative severity scales must have reasonably comparable meaning. Review their calibration on a labeled development sample before interpreting subtraction as disagreement.

## 8. Static and temporal disagreement

The following formulas define a transparent prototype. They are proxies for disagreement, not a formal logical proof of contradiction.

### Static under-acknowledgement

For an eligible topic with numerical evidence and an applicable narrative claim:

    S[j,t] = max(0, B[j,t] - A[j,t])

This focuses on numerical weakness that management's applicable claims under-acknowledge.

Example: numerical severity 0.85 and narrative acknowledged severity 0.20 give S = 0.65.

A narrative that is more pessimistic than the numbers may also contain useful information. Export the reverse gap separately:

    R[j,t] = max(0, A[j,t] - B[j,t])

Do not merge both directions without testing whether they mean the same thing for the prediction task.

### Temporal under-acknowledgement

For two comparable, available observations of the same topic:

    delta_B = B[j,t] - B[j,t_prev]
    delta_A = A[j,t] - A[j,t_prev]
    T[j,t] = min(1, max(0, delta_B - delta_A)) if delta_B > 0 else 0

This becomes positive when numerical severity worsens more than management's acknowledged severity changes.

Compare matched intervals. Record elapsed time and do not directly compare a one-year change with a three-year change. For irregular intervals, either restrict to a specified interval tolerance or implement and validate a rate-based version with explicit units.

A temporal score can remain positive even when management already acknowledges high absolute risk. That is why S and T must stay separately inspectable rather than being interpreted as identical signals.

## 9. Aggregate score and coverage

For each topic, first produce a topic score:

    c[j,t] = (alpha * S[j,t] + beta * T[j,t]) / (alpha + beta)

Use this formula only when both components are available. If temporal history is missing, use c = S and mark temporal availability as false. If the current topic is ineligible, c is unavailable.

Then aggregate across eligible topics:

    C[t] = sum(w[j] * c[j,t]) / sum(w[j])

The denominator includes eligible topics only. All weights are nonnegative. Begin with equal topic weights. Choose alpha and beta on validation data or specify them in advance; do not optimize them on the test set.

With bounded components and this weighted average, C lies in [0,1]. This boundedness applies to this MVP formulation, not automatically to the earlier raw geometric operator.

Always return coverage separately:

    coverage[t] = sum(weights of eligible topics) / sum(weights of configured topics)

If no topic is eligible, return C = null with coverage = 0. Do not output zero contradiction. Do not multiply C by low coverage and quietly make a poorly observed company look consistent.

A separate model-confidence estimate can be added if calibrated. Coverage alone is not confidence.

## 10. Worked example

Assume a reviewed, applicable liquidity claim and a matched one-year history:

| Quantity | Previous filing | Current filing |
| --- | --- | --- |
| Numerical liquidity severity B | 0.40 | 0.85 |
| Narrative acknowledged severity A | 0.20 | 0.25 |

    S = max(0, 0.85 - 0.25) = 0.60
    delta_B = 0.45
    delta_A = 0.05
    T = min(1, max(0, 0.45 - 0.05)) = 0.40

With alpha = beta = 1:

    c_liquidity = (0.60 + 0.40) / 2 = 0.50

Interpretation: the observed claims under-acknowledge numerical liquidity weakness, both currently and in the change over the interval. This is illustrative arithmetic, not a trained model result or a 50% bankruptcy probability.

If management explains a confirmed new credit facility, the claim review may reduce the apparent mismatch or mark the comparison uncertain. Evidence and context matter more than the arithmetic alone.

## 11. Auditor evidence

Keep auditor acknowledgement and warnings as separate features initially. Store the relevant text, date, topic, and extraction confidence.

Potential features include management-auditor disagreement and explicit auditor warnings. Auditor text can support interpretation, but it is not infallible and may be sparse or standardized.

Avoid an adaptive “trust” filter in version 1. A filter trained to reward agreement with our numerical score can create a circular definition of reliability. Introduce one only with a clear target, defensible assumptions, and an ablation against fixed or absent reliability weighting.

## 12. Output contract

One record per company and prediction cutoff should contain:

```json
{
  "company_id": "example_firm",
  "prediction_cutoff": "YYYY-MM-DD",
  "contradiction_score": 0.50,
  "coverage": 0.25,
  "score_version": "topic_gap_v1",
  "topics": {
    "liquidity": {
      "numerical_severity": 0.85,
      "narrative_acknowledged_severity": 0.25,
      "static_gap": 0.60,
      "temporal_gap": 0.40,
      "reverse_gap": 0.0,
      "topic_score": 0.50,
      "temporal_available": true,
      "evidence_sentence_ids": ["filing_123_sentence_45"],
      "limitations": []
    }
  },
  "missing_topics": ["leverage", "profitability", "growth"]
}
```

The coverage value assumes four equally weighted configured topics with only liquidity eligible. Store source sentences, financial values, and their dates in an associated evidence table. Avoid copying full filings into every output record.

## 13. Integration into the codebase

The friend or coding assistant should first inspect the repository and identify the actual dataset loader, split logic, preprocessing, model interfaces, and evaluation scripts. The names below describe new responsibilities, not verified existing paths.

Suggested modules:

- `financial_topic_scores`: fit training-only numerical transforms and calculate B.
- `narrative_claims`: retrieve and classify applicable claims, produce A and evidence.
- `contradiction_score`: compute component gaps, aggregation, coverage, and histories.
- `contradiction_evaluation`: run baselines, ablations, and behavioral checks.

Suggested fit/transform interface:

```python
scorer.fit(train_records, claim_annotations=None)
train_features = scorer.transform(train_records)
val_features = scorer.transform(val_records)
test_features = scorer.transform(test_records)
```

`transform` must not refit any scaler, classifier, topic weights, or thresholds. Historical observations may be consumed only when available before each record's cutoff. Historical state must be keyed by company and processed chronologically.

Integration sequence:

1. Preserve a reproducible MABAD baseline and record its configuration and metrics.
2. Export score features keyed to existing samples and verify joins preserve row count and labels.
3. Run a simple predictor using score features to check whether they carry signal.
4. Test a controlled late-fusion model using MABAD output/representation plus C, S, T, reverse gaps, and availability features.
5. Modify the neural architecture only after that comparison demonstrates useful signal.

If a second-stage predictor consumes MABAD predictions, train it with out-of-fold or otherwise held-out baseline predictions. In-sample predictions from a baseline trained on the same rows can overstate the stack's performance.

Keep the ordinary financial risk signal alongside contradiction. Do not replace bankruptcy risk with C or force high C to mean bankruptcy.

## 14. How it could help

### Prediction

The score could identify a pattern that ordinary multimodal fusion does not capture strongly enough: worsening finances with limited or inconsistent narrative acknowledgement. This is a hypothesis to test, not a guaranteed benefit.

### Explanation

For each flagged company, we can show the financial topic, contributing ratios, management sentence, observed mismatch, and historical comparison. This provides a concrete explanation of the feature's construction.

### Review prioritization

A review interface could distinguish “high numerical distress with acknowledgement” from “high numerical distress with unsupported reassurance.” The contradiction score can rank cases for closer review without alleging dishonesty.

### Research contribution

Component ablations and annotated cases can establish whether the explicit feature adds information beyond MABAD. They are more informative than reporting a new algorithm name alone.

## 15. Pros and cons

| Pros | Cons / failure modes |
| --- | --- |
| Explicit, inspectable topic contributions | Severity scales may not be directly comparable without calibration |
| Straightforward first implementation and debugging | Claim extraction, negation, and timing can be difficult |
| Can use static scores when history is unavailable | Temporal scores need repeated, comparable observations |
| Shows evidence rather than only a final risk value | Low cash or high leverage may be normal for some sectors |
| Separates contradiction from ordinary distress | MABAD may already learn the same information, leaving little incremental gain |
| Supports targeted ablations | Adding features may overfit a dataset with very few bankrupt firms |
| Can expose missingness explicitly | Reporting style and boilerplate can drive false disagreement |
| Makes a simpler comparator for geometric research | The MVP may offer explanation value without being a novel algorithm |

## 16. Validation plan

### Behavioral tests

- Weak numbers + explicit acknowledgement: low static gap, but the overall bankruptcy model can remain high risk.
- Weak numbers + applicable unsupported reassurance: higher static gap.
- Same weak numbers + a verified mitigating explanation: review or reduce the mismatch instead of blindly flagging it.
- No relevant management claim: unavailable topic, not an optimistic claim.
- Worsening numbers + matching increase in acknowledged severity: lower temporal gap.
- Worsening numbers + little acknowledgement change: higher temporal gap.
- Financial improvement: zero worsening-specific temporal gap.
- Missing ratios/history: explicit missingness, finite eligible outputs, no fabricated observation.
- Future filing added to the data: it must not change a previously computed historical score.

### Predictive comparisons

Use the same chronological splits and comparable tuning budgets:

1. MABAD baseline.
2. MABAD plus static score and its availability/coverage features.
3. MABAD plus static and temporal components.
4. MABAD plus numerical severity and narrative severity separately, without their gap.
5. MABAD plus omission/coverage features only.
6. Optional geometric operator variants, one component at a time.

Comparison 4 checks whether gains come from disagreement itself rather than simply adding two useful severity features. Comparison 5 checks whether missingness explains the apparent gain.

Primary reporting: average precision and recall at a fixed review budget, with ROC-AUC as a complementary metric. Evaluate Brier score/calibration only for outputs that represent probabilities. Report uncertainty, preferably using company-grouped resampling where appropriate.

Evaluate claim extraction on held-out human annotations, not only bankruptcy labels. For contradiction-heavy slices, use independently reviewed definitions; selecting cases solely because our score is high creates a circular evaluation.

## 17. Acceptance criteria and team work

Before calling the extension successful:

- Data joins, cutoff dates, and training-only transforms pass checks.
- Evidence extraction behaves correctly on a reviewed sample.
- Scores are reproducible, bounded as specified, and have explicit missingness.
- The baseline comparison uses an untouched future test period.
- Results show whether gains persist beyond added severity and coverage features.
- Explanations identify source evidence and acknowledge uncertainty.

Suggested ownership: one teammate owns data/schema and date alignment, one owns claim extraction and annotations, and one owns score construction, integration, and experiments. All should review ambiguous financial claims together.

## 18. Instructions for the friend / coding assistant

Treat this document as a proposal to implement and test, not proof of an established invention. Inspect the code and data first. Report which required fields and histories exist. Implement Track A with configurable topic mappings and explicit missingness. Preserve the baseline and add tests for leakage, score behavior, and row alignment. Do not invent experimental gains, quarterly histories, financial ground truth, or the exact architecture of existing repository modules.

If necessary inputs are absent, implement the supported static subset and document what prevents temporal scoring. Do not substitute generic sentiment or raw embedding distance without naming it as a weaker baseline.

## 19. Project material behind this handoff

- The team's updated MABAD project report: baseline setup and repository benchmark interpretation.
- `Geodesic Curvature Lie Operator for NNCE.pdf`: the earlier experimental geometric design.
- `Contradiction_Score_Professor_Explained.pptx`: the professor-facing concept presentation.

This document deliberately introduces a simpler MVP before the geometric research version. Team agreement on the chosen track should precede claims that the algorithm in the presentation has been implemented.
