# Preliminary Literature Scan: MABAD Aggregation Research

**Date:** 2026-10-09  
**Status:** Initial scoping scan, not a systematic review and not a novelty clearance.

## What the initial scan suggests

The general idea of comparing maximum, mean, median, weighted, or other score-fusion functions is **not novel by itself**. Score-level fusion is an established family of techniques in anomaly detection, and recent work explicitly compares max and mean aggregation. Multimodal financial-distress prediction also already contains many attention/gating/fusion architectures. Therefore, a paper that only swaps `max` for `mean` on one dataset is unlikely to support a strong novelty claim.

A potentially more defensible question would need to be more specific to MABAD's setting: heterogeneous anomaly-distance scales, severe label imbalance, missing/unequal modality availability, and the role of the original max rule within its aligned anomaly-distance representation. Any new method must be justified by a principled design, validated against appropriate baselines, and evaluated without test-set selection.

## Relevant starting points

1. **MABAD original repository / paper context** — describes the target baseline, severe class imbalance, missing financial data, sparse text modalities, and chronological split. This is the main baseline to distinguish from, not evidence that a new aggregation is novel.  
   https://github.com/iit-Demokritos/MABADforBankruptcyPrediction

2. **Multimodal Generative Models for Bankruptcy Prediction Using Textual Data** (Mancisidor & Aas, 2022 preprint) — multimodal bankruptcy prediction and missing text modality context.  
   https://arxiv.org/abs/2211.08405

3. **From Numbers to Words: Multi-Modal Bankruptcy Prediction Using the ECL Dataset** (Arno et al., 2023, FinNLP/ACL Anthology) — evaluates classical and neural multimodal bankruptcy prediction and discusses complementary modalities and target limitations.  
   https://aclanthology.org/2023.finnlp-2.2/

4. **Multi-modal bankruptcy risk prediction for listed companies via large language models** (Information Sciences, 2026) — a recent example of multimodal bankruptcy risk prediction with adaptive modality fusion.  
   https://www.sciencedirect.com/science/article/pii/S002002552600318X

5. **Disentangling multi-modal fusion for financial distress prediction with multi-level attention mechanisms** (Applied Soft Computing, 2026) — another recent financial-distress work that explicitly models intra- and inter-modal fusion and modality differences/commonalities.  
   https://www.sciencedirect.com/science/article/pii/S1568494626005338

6. **Unsupervised Anomaly Detection Framework for Multimodal Data in Industrial Control Systems** — describes late fusion of per-modality anomaly scores by arithmetic mean and contrasts score fusion with joint feature fusion. This demonstrates that simple mean score fusion is an established baseline in multimodal anomaly detection.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC13306929/

7. **FMFR: Feature-level multi-stage fusion and remapping for multi-modal industrial anomaly detection** (Journal of Computational Design and Engineering, 2026) — explicitly reports an aggregation ablation including mean and max and finds their performance differs by task/metric.  
   https://academic.oup.com/jcde/article/13/3/233/8502145

8. **G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection** (ICCV 2025) — discusses existing score addition/max-selection strategies and motivates considering additional geometric information. It is adjacent anomaly-detection work, not bankruptcy evidence.  
   https://openaccess.thecvf.com/content/ICCV2025/papers/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.pdf

## Implications for our project

- **Do not claim novelty for mean/max/power-mean aggregation alone.** These are standard operators.
- The first ablation is a diagnostic study: it tests whether the baseline result depends strongly on its max rule and identifies a better-motivated research question.
- A stronger potential contribution would need a new, principled score-calibration or aggregation design specifically suited to MABAD's modality-distance scales and rare-event evaluation, plus rigorous comparisons to simple baselines.
- Because financial distance is nearly constant for most rows and has a much different scale/distribution than the text distances, raw mean can behave in ways that reflect scale quirks rather than an intended equal-modality fusion. The method must explicitly address and test this.
- Search further for score calibration, rank/CDF normalization, generalized means, robust aggregation, OWA/Choquet integrals, and anomaly-score fusion under class imbalance before finalizing the novelty statement.
- This scan is not exhaustive. Verify publication metadata, read full papers where accessible, and search scholarly databases systematically before claiming a gap or drafting a patent claim.
