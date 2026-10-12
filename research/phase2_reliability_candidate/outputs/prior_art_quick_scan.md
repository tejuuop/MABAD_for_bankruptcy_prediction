# Quick public prior-art scan (not exhaustive; checked 2026-10-10)

This is a preliminary screening, not a legal opinion and not a full patent search.

1. **MABAD original work / code** — “A Multimodal Alignment-Based Anomaly Detection Method for Bankruptcy Prediction.” The project’s own base method already combines financial and annual-report text modalities for bankruptcy anomaly detection, so the disclosure must not claim the base architecture or application as new.
   - Paper record: https://dice.iit.demokritos.gr/
   - Public code repository: https://github.com/iit-Demokritos/MABADforBankruptcyPrediction

2. **Generic score normalization and weighted anomaly-detector fusion** — Fatemifar et al., “Developing a generic framework for anomaly detection,” *Pattern Recognition* (2022). The abstract describes score normalization and weighted fusion of one-class classifiers. This overlaps with any broad claim that merely normalizes anomaly scores and averages them.
   - https://doi.org/10.1016/j.patcog.2021.12.006
   - Search result/article record: https://www.sciencedirect.com/science/article/abs/pii/S0031320321006762

3. **Multimodal anomaly-score normalization and weighted fusion in another domain** — a recent Scientific Reports article describes normalizing per-modality anomaly scores and combining them using modality weights. This reinforces that generic “normalize then weighted-fuse” is unlikely to be a sufficient novelty claim by itself.
   - https://doi.org/10.1038/s41598-026-52894-w

## Implication for this project
Do **not** claim that percentile normalization, weighted averaging, or IQR weighting is globally new. The defensible project contribution at this stage is a domain-specific, reproducible empirical investigation of how modality-distance distribution mismatch affects MABAD-style bankruptcy ranking, with validation-only calibration and leakage-aware evaluation. A narrower inventive claim would need a more distinctive technical mechanism and a broader patent search.

Before an invention disclosure is filed, ask the university’s IP/technology-transfer office whether this form is intended to disclose research in progress or an actual invention, and whether public disclosure/submission timing affects rights. Do not delay the academic submission, but avoid asserting patentability.
