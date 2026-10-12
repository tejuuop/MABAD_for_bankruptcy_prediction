# Targeted Literature and Novelty Assessment

## Search scope and limitation

This is a targeted, non-exhaustive review conducted on 2026-10-12. It covers the method families relevant to the repository: classifier and score aggregation, robust/reliability-aware multimodal fusion, multimodal bankruptcy prediction, and numerical–narrative disclosure consistency. It is not a systematic review, legal prior-art search, or patentability opinion.

## Verified references

| Area | Reference | Relevance and limitation |
|---|---|---|
| MABAD baseline | Sideras, A., Bougiatiotis, K., Zavitsanos, E., Paliouras, G., & Vouros, G. (2025). *A Multimodal Alignment-Based Anomaly Detection Method for Bankruptcy Prediction*. ICAIF 2025, pp. 53–61. DOI: [10.1145/3768292.3770380](https://doi.org/10.1145/3768292.3770380). | Establishes the base MABAD method and application. Any contribution here must be an extension or analysis of MABAD, not a claim to multimodal anomaly-based bankruptcy prediction generally. |
| Classifier fusion | Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). *On Combining Classifiers*. IEEE TPAMI, 20(3), 226–239. DOI: [10.1109/34.667881](https://doi.org/10.1109/34.667881). | Fixed algebraic combinations of multiple classifiers/scores are established. It does not establish a MABAD-specific bankruptcy result. |
| Power-mean ensemble family | Hassan, M. F., Abdel-Qader, I., & Bazuin, B. (2021). *A new method for ensemble combination based on adaptive decision making*. Knowledge-Based Systems, 233, 107544. DOI: [10.1016/j.knosys.2021.107544](https://doi.org/10.1016/j.knosys.2021.107544). | Treats power/generalized-mean fusion as an established family. A raw p=4 rule is not, by itself, a novel operator. |
| Reliability-aware multimodal fusion | Zhang, Q., Wei, Y., Han, Z., et al. (2026). *Multimodal Fusion on Low-quality Data: A Comprehensive Survey*. Information Fusion, 104437. DOI: [10.1016/j.inffus.2026.104437](https://doi.org/10.1016/j.inffus.2026.104437). | Documents noisy, incomplete, imbalanced, and quality-varying modality settings. It makes broad “reliability-aware fusion” novelty untenable without a specific new mechanism and validation. |
| Quality-aware fusion | Zhu, Y., Jiang, Y., Jiang, G., et al. (2026). *QA-MoE: Towards a Continuous Reliability Spectrum with Quality-Aware Mixture of Experts for Robust Multimodal Sentiment Analysis*. ACL 2026. DOI: [10.18653/v1/2026.acl-long.1461](https://doi.org/10.18653/v1/2026.acl-long.1461). | Example of explicit modality-reliability estimation and adaptive routing. Different domain/task, but it shows that generic uncertainty-aware weighting is established. |
| Multimodal bankruptcy data and baselines | Arno, H., Mulier, K., Baeck, J., & Demeester, T. (2023). *From Numbers to Words: Multi-Modal Bankruptcy Prediction Using the ECL Dataset*. FinNLP 2023, pp. 11–21. DOI: [10.18653/v1/2023.finnlp-2.2](https://doi.org/10.18653/v1/2023.finnlp-2.2). | Establishes that numerical and narrative 10-K data can be combined for bankruptcy research. It does not examine the MABAD-distance power-mean mechanism. |
| Deep financial-text fusion | Islam, M. S., et al. (2025). *A Dual-Encoder Approach to Bankruptcy Prediction With Financial Ratios and Corporate Disclosures*. IEEE Open Journal of the Computer Society, 6, 1896–1907. DOI: [10.1109/OJCS.2025.3629748](https://doi.org/10.1109/OJCS.2025.3629748). | Uses a gated multimodal deep architecture. This makes a broad claim of combining ratios and disclosures non-novel; the mechanism and evidence would need to be distinct. |
| Narrative disclosure and bankruptcy | Tennyson, B. M., Ingram, R. W., & Dugan, M. T. (1990). *Assessing the Information Content of Narrative Disclosure in Explaining Bankruptcy*. Journal of Business Finance & Accounting, 17(3), 391–410. DOI: [10.1111/j.1468-5957.1990.tb01193.x](https://doi.org/10.1111/j.1468-5957.1990.tb01193.x). | Longstanding evidence that narratives and financial distress can be studied together. It cautions against calling the broad numerical–narrative link novel. |
| Narrative performance relation | Smith, M., & Taffler, R. (1992). *The chairman's statement and corporate financial performance*. Accounting & Finance, 32(2), 75–90. DOI: [10.1111/j.1467-629X.1992.tb00187.x](https://doi.org/10.1111/j.1467-629X.1992.tb00187.x). | Studies the connection between managerial narrative properties and performance, including liquidity. It is related background for disclosure inconsistency. |
| Disclosure complexity and performance | de Souza, J. A. S., Rissatti, J. C., Rover, S., & Borba, J. A. (2019). *The linguistic complexities of narrative accounting disclosure on financial statements: An analysis based on readability characteristics*. Research in International Business and Finance, 48, 59–74. DOI: [10.1016/j.ribaf.2018.12.008](https://doi.org/10.1016/j.ribaf.2018.12.008). | Shows an established literature on narrative obfuscation/performance relations, but does not validate the repository’s topic-level acknowledgement rules. |
| Tone inconsistency | Pouryousof, A., Nassirzadeh, F., & Askarany, D. (2023). *Inconsistency in Managers’ Disclosure Tone: The Signalling Perspective*. Risks, 11(12), 205. DOI: [10.3390/risks11120205](https://doi.org/10.3390/risks11120205). | Directly relevant evidence that disclosure inconsistency/tone has established research precedent. Tone residuals are not the same as topic-specific under-acknowledgement. |
| Impression management close to bankruptcy | Camacho-Miñano, M. M., & Alonso Moreno, M. M. (2024). *Impression management in bankruptcy: An analysis of the narratives in listed companies*. Revista de Contabilidad – Spanish Accounting Review, 27(1), 160–173. DOI: [10.6018/rcsar.512151](https://doi.org/10.6018/rcsar.512151). | Relevant to management narratives near bankruptcy. It reinforces the need for careful claims rather than a novelty assertion for “narratives may be optimistic.” |

## What is already established

1. Multimodal bankruptcy prediction with structured and textual corporate disclosures.
2. Fixed and learned aggregation/fusion of multiple model or modality outputs.
3. Generalized/power-mean families of score aggregation.
4. Reliability-aware handling of missing, unequal-quality, and noisy modalities.
5. Links between disclosure style/tone, financial performance, distress, and impression management.

## What the repository could distinctively study

A narrow empirical contribution remains plausible but unproven: **within a fixed MABAD anomaly-distance system, does a pre-specified generalized mean yield more useful rare-event ranking than hard maximum aggregation under a leakage-aware, group-preserving protocol?** The novelty would be in the MABAD-specific diagnosis, transparent ablation, and reproducibility package—not in inventing the generalized mean itself.

The proposed Track A construction would need a more distinctive mechanism than “financial and text features disagree.” Its potentially distinctive ingredients are topic matching, explicit numerical severity, evidence retention, and a strictly missing-not-optimistic rule. Those are not yet validated because the claim extractor and annotation provenance are inadequate.

## Novelty classification

| Proposed claim | Classification now | Reason |
|---|---|---|
| “Power mean aggregation is new.” | Unsupported. | Established generalized-mean/ensemble-fusion family. |
| “Reliability-aware multimodal fusion is new.” | Unsupported. | Broadly established literature. |
| “Numbers and disclosures can be combined for bankruptcy prediction.” | Unsupported. | Established multimodal bankruptcy literature. |
| “MABAD-specific fixed p=4 aggregation improves bankruptcy prediction.” | Inconclusive. | Promising validation point estimate, but CI includes zero and no independent confirmation exists. |
| “Topic-level numerical–narrative under-acknowledgement is a validated incremental bankruptcy feature.” | Unsupported. | Current narrative labels/proxies lack verified human validation and fair predictive evaluation. |
| “An auditable diagnostic layer can expose evidence coverage and disagreement without claiming prediction gain.” | Potentially useful engineering contribution; research value unproven. | Interpretability/evidence retention are present, but user studies or validated annotations are absent. |

## Search extensions required before submission

- Google Scholar / Scopus / Web of Science: `("power mean" OR "generalized mean") anomaly score fusion bankruptcy`.
- IEEE Xplore / ACM DL: `multimodal anomaly detection late fusion max mean generalized mean`.
- SSRN / accounting journals: `MD&A tone financial performance inconsistency bankruptcy disclosure`.
- Patent databases (USPTO, Espacenet, WIPO): `financial anomaly score multimodal fusion`, `bankruptcy prediction text financial ratios`, and `narrative financial contradiction scoring`.

No novelty or patentability conclusion should be drawn until that broader, professionally reviewed search is complete.
