# Phase 0A Completeness Summary

- Rows: 39041
- Positives: 192
- Companies: 4855
- Date range: 1999-12-31 to 2021-10-31

## Signal Summary

| signal | pos mean | neg mean | diff 95% CI | best AP direction | best AP | best Recall@50 | best Recall@100 |
|---|---:|---:|---:|---|---:|---:|---:|
| financial_nonnull_count | 197.125 | 208.430 | [-20.722, -1.075] | lower values riskier | 0.0053 | 0.0052 | 0.0104 |
| financial_missing_pct | 0.980 | 0.979 | [0.000, 0.002] | higher values riskier | 0.0053 | 0.0052 | 0.0104 |
| mda_char_count | 78466.651 | 72093.809 | [713.039, 12196.119] | higher values riskier | 0.0059 | 0.0000 | 0.0000 |
| auditor_char_count | 4197.031 | 3967.248 | [-145.197, 660.893] | higher values riskier | 0.0059 | 0.0052 | 0.0052 |
