# Phase 0 Decision

Decision: D. Useful diagnostic signal, but insufficient evidence yet for a final new scorer.

Rationale:

- Saved-model inference recovered modality distances without retraining. `d_max` gives AP 0.0989, AUC 0.9003, Recall@50 0.1364, and Recall@100 0.2424 on validation+test.
- The MD&A distance is the strongest individual modality distance: AP 0.0951, AUC 0.8796.
- Disagreement is a real diagnostic signal but weaker than the core distances: disagreement standard deviation gives AP 0.0636, AUC 0.7868.
- Completeness/missingness alone is not a useful bankruptcy ranking signal, although financial completeness differs slightly between bankrupt and non-bankrupt filings.
- The next step should be a validation-only post-hoc reliability experiment, not a new architecture.

Recommended next experiment:

Fit a tiny validation-only post-hoc scorer using only transparent features: `d_financial`, `d_mda`, `d_auditor`, `d_max`, normalized disagreement range, financial missing percentage, MD&A length, auditor length, and dominant modality. Evaluate once on test against the historical MABAD/XGBoost rows.
