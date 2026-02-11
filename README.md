

# comparing 10 CNN and Transformer backbones across 9 heterogeneous datasets (medical, natural images, remote sensing, fine-grained).

All models are trained under an identical protocol (same optimizer, LR schedule, epochs, augmentations, resolution), removing tuning and pretraining bias.

Given your setup (10 CNN/Transformer backbones × 9 heterogeneous datasets, identical training protocol, no pretraining/tuning), you should structure the paper around (i) clearly scoped research questions, (ii) a fairness contract for the protocol, and (iii) analyses that explain *when/why* backbones win or fail across domains—not just average accuracy. Large-scale backbone comparisons are a known paper pattern in vision, so reviewers will look for strong baselines, robustness checks, and careful validity discussion.[^1][^2]

## Evaluation-paper outline (tailored)

### Title / Abstract / Keywords

1. Title: “A Controlled Evaluation of CNN vs Transformer Backbones Across Heterogeneous Vision Domains” (make “controlled/identical protocol/no pretraining” explicit).[^3]
2. Abstract: task suite + protocol constraint (identical optimizer/schedule/epochs/augs/resolution), headline results (mean + dispersion), and actionable recommendations by domain type.[^3]
3. Keywords: backbone comparison, controlled evaluation, domain shift, CNN, ViT.[^4]

### 1. Introduction

1. Motivation: backbone choice drives performance and cost; heterogeneous datasets create implicit distribution shift that makes “one-dataset” conclusions unreliable.[^2][^5]
2. Problem: current practice confounds architecture with tuning and pretraining; you isolate architecture under a single protocol to reduce these biases.[^1]
3. Contributions (bullets)

- Controlled benchmark: 10 backbones × 9 datasets under a single training recipe (no pretraining, no per-dataset tuning).[^1]
- Cross-domain analysis: ranking stability, variance, and domain-specific winners/losers.[^1]
- Practical guidance: which backbones are robust under “one protocol fits many.”[^2]

4. Summary of findings (1 paragraph, directional).[^3]

### 2. Benchmark scope and research questions

1. Benchmark axes: architecture family (CNN/Transformer), dataset domain (medical/natural/remote sensing/fine-grained), dataset size/label space if relevant.[^5]
2. Research questions (examples you can adapt)

- RQ1: Under a single standardized protocol, which backbones perform best on average and per domain?[^1]
- RQ2: How stable are backbone rankings across datasets (and across seeds)?[^1]
- RQ3: Are Transformers more sensitive to protocol choices (even when fixed) than CNNs, e.g., data regime or compute regime?[^1]
- RQ4: What is the accuracy–compute frontier for each backbone under identical training budget?[^2]

3. Claims table (1 page max): each claim mapped to a figure/table you will produce (helps both writing and artifact review).[^6][^7]

### 3. Related work

1. Backbone benchmarking and “battle of backbones” style comparisons; clarify how yours differs (e.g., no pretraining, controlled recipe, heterogeneous domains).[^1]
2. Domain shift / robustness motivation for multi-domain evaluation (why cross-dataset heterogeneity matters).[^5]
3. Notes on confounds: pretraining and tuning can dominate outcomes; your design intentionally removes these factors (but discuss trade-offs).[^1]

### 4. Experimental design (the centerpiece)

#### 4.1 Models (10 backbones)

- Define each backbone, parameter count, FLOPs (at the chosen resolution), and any architectural constraints that interact with the fixed protocol.[^1]


#### 4.2 Datasets (9 heterogeneous)

- For each dataset: domain type, task type (classification? multi-label?), number of classes, sample size, image characteristics (channels, typical resolution), and split strategy.[^8]
- Explain any harmonization required (e.g., resizing to a single resolution) and why that choice is defensible.[^2]


#### 4.3 The “fairness contract” (identical protocol)

1. Training recipe (your fixed optimizer, LR schedule, epochs, augmentations, resolution) as a fully specified contract.[^2]
2. Justification: you are measuring “out-of-the-box under one recipe” performance, not “best possible with tuning.”[^1]
3. Budget statement: identical epochs doesn’t necessarily mean identical compute—state whether you equalize epochs, wall-clock, or FLOPs, and discuss implications.[^2]

#### 4.4 Evaluation methodology

1. Metrics: primary metric (e.g., top-1 accuracy / AUC depending on dataset), secondary metrics (calibration, macro-F1 for imbalance, etc.) with rationale.[^2]
2. Statistical protocol: number of seeds, CI computation, and how you aggregate across datasets (mean rank, average normalized score, etc.).[^2]
3. Implementation details that affect reproducibility: frameworks/versions, mixed precision, deterministic settings, and hardware.[^7][^8]

### 5. Results (organized by RQs)

#### 5.1 Main performance (RQ1)

- Table: backbone × dataset performance, plus domain-group aggregates.[^1]
- Include dispersion (std/CI) because identical protocol can still yield variance.[^2]


#### 5.2 Ranking stability (RQ2)

- Rank correlation across datasets; identify “universally good” vs “domain specialists.”[^1]
- Highlight failure cases (e.g., a backbone that is top-3 on natural images but collapses on medical).[^5]


#### 5.3 Sensitivity analyses (supports internal validity)

Even if you disallow per-dataset tuning, you should still test whether conclusions are fragile to *small* global changes:

- Protocol sensitivity: a small set of alternate global recipes (e.g., two LR schedules, two augmentation strengths) applied uniformly to all datasets/backbones.[^2]
- Data regime sensitivity: subsample curves on a subset of datasets (small/medium/large) to see which family degrades gracefully.[^1]


#### 5.4 Efficiency and scaling (RQ4)

- Accuracy vs FLOPs/params/latency at fixed resolution; show Pareto frontiers.[^2]


#### 5.5 Error and slice analysis (actionable)

- Per-class or per-slice breakdown (especially for fine-grained and medical), to avoid “average score hides systematic failure.”[^2]


### 6. Discussion

1. Practical recommendations: “If you must use one fixed recipe across varied datasets, choose backbone X/Y; avoid Z in regime R.”[^2]
2. Interpretation: connect differences to dataset regime/shift intuitions, without over-claiming causality.[^5]
3. Relationship to pretrained reality: explain how your results should (and should not) be extrapolated to pretrained backbones.[^1]

### 7. Threats to validity and limitations (be explicit)

1. Construct validity: fixed resolution/augmentations may advantage some backbones; metric choice differs across domains.[^2]
2. Internal validity: identical epochs ≠ identical compute; implementation differences; seed variance.[^7][^2]
3. External validity: “no pretraining” may not reflect typical practice; dataset selection may bias conclusions.[^1]
4. Recommendation: frame conclusions as “controlled under recipe R” and provide guidance on how to adapt.[^1]

### 8. Reproducibility / artifacts (often a differentiator)

1. Public repo with configs, exact commands, and scripts to regenerate every figure/table.[^6][^7]
2. Claim-to-evidence index: each claim references a table/figure + script + config hash.[^7]

### 9. Conclusion

- Answer each RQ succinctly and restate the decision guidance your evaluation enables.[^2]

If you share the task type(s) across the 9 datasets (all single-label classification, or mixed with multilabel/segmentation?) and whether you’ll equalize compute (FLOPs/time) or stick with equal epochs, I can tighten Sections 4–7 to match reviewer expectations for “fair comparison” in vision benchmarking.
 

<div align="center">⁂</div>

[^1]: https://arxiv.org/pdf/2310.19909.pdf

[^2]: https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf

[^3]: https://bibliotecnica.upc.edu/sites/default/files/pagines_generals/investigadors/ieee_authorship_presentation_11152023.pdf

[^4]: https://www.acm.org/binaries/content/assets/publications/taps/acm_layout_submission_template.pdf

[^5]: https://arxiv.org/pdf/2404.00897.pdf

[^6]: https://www.acm.org/publications/policies/artifact-review-badging

[^7]: https://vahldiek.github.io/files/acmrep-5yeareurosysae.pdf

[^8]: https://sigsim.acm.org/conf/pads/2023/blog/artifact-evaluation/

[^9]: outline.md

[^10]: https://www.sciencedirect.com/science/article/pii/S0950584925002150

[^11]: https://developers.google.com/machine-learning/crash-course/fairness/mitigating-bias

[^12]: https://cycle.io/learn/bias-and-fairness-in-machine-learning

[^13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11104322/

[^14]: https://www.nature.com/articles/s41598-024-64210-5

[^15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12289721/

[^16]: https://www.biorxiv.org/content/10.1101/682997v2.full.pdf

[^17]: https://www.cs.purdue.edu/homes/lintan/publications/fairness-neurips21.pdf

[^18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12286580/

[^19]: https://arxiv.org/html/2401.13391v1

[^20]: https://wandb.ai/wandb_fc/genai-research/reports/Transfer-learning-versus-fine-tuning--VmlldzoxNDQxOTM3OQ

[^21]: https://www.nature.com/articles/s41598-022-06484-1

[^22]: https://dl.acm.org/doi/full/10.1145/3551390

[^23]: https://openaccess.thecvf.com/content/WACV2024W/Pretrain/papers/Seth_Does_the_Fairness_of_Your_Pre-Training_Hold_Up_Examining_the_WACVW_2024_paper.pdf

