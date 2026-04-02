# Analysis Figures — Caption Reference

**Thesis:** Few-Shot Learning Paradigms for Bird Species Classification under Data Scarcity
**Location:** `results/birds20/analysis/`
**Total figures:** 38 PNG files

Figures are grouped by theme. All experiments use a 5-way episodic protocol evaluated over 600 episodes with 95% Wilson confidence intervals. Two dataset splits are used throughout:
- **B15/N5** — 15 base classes / 5 novel classes
- **B10/N10** — 10 base classes / 10 novel classes

---

## 1. Thesis Master Figure

### `thesis_summary_figure.png`
A single-panel master overview of the thesis results. Combines the best accuracy per method and split into a unified comparison, intended as the primary summary figure for the thesis document. Includes supervised baselines (Full-20, Base-15), NCC, ProtoNet (Euclidean), MatchingNet, and RelationNet at 5-shot.

---

## 2. Dataset Overview

### `dataset_class_distribution.png`
Bar chart showing the number of training images per bird species across all 20 classes. Bars are colour-coded by split assignment (blue = base classes, orange = novel classes). Illustrates the near-uniform distribution (~155–166 images per class) and confirms that data scarcity is artificially imposed through the episodic K-shot protocol rather than natural class imbalance.

---

## 3. Supervised Baseline Panels

### `full20_results_panel.png`
Four-panel summary of the supervised ResNet-50 baseline trained on all 20 bird species:
- **Top-left:** Confusion matrix (20×20) on the full test set
- **Top-right:** Per-class F1 score bar chart
- **Bottom-left:** Training and validation loss curves across epochs
- **Bottom-right:** Training and validation accuracy curves

Achieves 99.00% top-1 accuracy — the practical upper bound for this dataset.

### `base15_results_panel.png`
Four-panel summary of the ResNet-50 baseline trained on the 15 base classes only:
- **Top-left:** Confusion matrix (15×15) on base-class test set
- **Top-right:** Per-class F1 scores for all 15 base species
- **Bottom-left/right:** Training and validation loss/accuracy curves

Achieves 100.00% accuracy on base classes — confirms the backbone learns near-perfect separation for seen classes before novel class evaluation.

---

## 4. ProtoNet (Prototypical Networks) — Per-Shot Panels

### `protonet_1shot_euc_panel.png`
Three-panel result summary for ProtoNet with Euclidean distance at K=1 shot (B15/N5 split):
- Confusion matrix over 600 episodes
- Per-class accuracy bar chart for 5 novel species
- t-SNE feature embedding coloured by novel class

Accuracy: **94.06%** ± 0.68% (95% CI).

### `protonet_3shot_euc_panel.png`
Same three-panel layout for ProtoNet Euclidean at K=3 shot (B15/N5).
Accuracy: **99.18%** ± 0.26%. Demonstrates near-saturation at 3-shot.

### `protonet_5shot_euc_panel.png`
Same layout for ProtoNet Euclidean at K=5 shot (B15/N5).
Accuracy: **99.80%** ± 0.10%. Best result in the entire study — matches and marginally exceeds the supervised Full-20 baseline.

### `protonet_euc_results_panel.png`
Aggregated panel showing all three shot counts (1, 3, 5) for ProtoNet Euclidean side-by-side. Visualises the diminishing returns pattern: large gain from 1→3 shot (+5.12 pp), minimal gain from 3→5 shot (+0.62 pp).

---

## 5. ProtoNet Ablation — Distance Metric

### `protonet_distance_ablation.png`
Side-by-side comparison of Euclidean vs Manhattan distance for ProtoNet across all shot counts and both splits. Shows that Euclidean distance consistently outperforms Manhattan by 6–15 percentage points across all configurations. Key finding: ResNet-50 avgpool features are better separated under L2 than L1 norm.

---

## 6. NCC vs ProtoNet

### `ncc_vs_protonet.png`
Direct comparison of NCC (Nearest Centroid Classifier, frozen backbone) vs ProtoNet (episodically trained backbone) across K=1, 3, 5 shots using Euclidean distance on both splits. Quantifies the contribution of episodic training beyond feature quality: ProtoNet outperforms NCC by ~1–6 pp, with the largest gap at 1-shot.

### `analysis_ncc_ablation.png`
Ablation study for NCC comparing Euclidean vs Manhattan distance metrics across all shot counts and both splits. Mirrors the ProtoNet distance ablation. Euclidean leads by 3–10 pp, confirming the distance metric finding is method-agnostic.

---

## 7. Research Question Visualisations (RQ1–RQ7)

### `rq1_protonet_vs_ncc.png`
**RQ1 — Does episodic training add value beyond a strong pre-trained backbone?**
Paired bar chart comparing ProtoNet (episodic) and NCC (frozen) at each shot count on B15/N5. Both use the same ResNet-50 architecture. The gap (ProtoNet − NCC) is labelled on each bar. Conclusion: yes, episodic training adds 1–6 pp, most meaningfully at 1-shot.

### `rq2_base_diversity_effect.png`
**RQ2 — How does base-class set size affect novel class generalisation?**
Compares B15/N5 vs B10/N10 accuracy for all FSL methods across all shot counts. The B15 split consistently outperforms B10 by 3–7 pp, showing that richer base-class diversity during backbone training improves novel class feature quality.

### `rq3_representation_vs_learning.png`
**RQ3 — Is performance driven by representation quality or the learning mechanism?**
Contrasts frozen NCC (representation only) against ProtoNet (learned episodic), MatchingNet (attention), and RelationNet (MLP comparator). Shows that a strong ResNet-50 backbone with simple nearest-centroid matching (NCC) already outperforms complex learned methods (MatchingNet, RelationNet), highlighting the dominance of representation quality.

### `rq4_shot_count_effect.png`
**RQ4 — How does the number of support shots affect performance?**
Line plots of accuracy vs K (1, 3, 5 shot) for all methods on both splits. Visualises the saturation pattern: steep gain from K=1 to K=3, near-flat from K=3 to K=5. Annotated with per-interval gains in percentage points.

### `rq5_distance_metric.png`
**RQ5 — How sensitive are metric-based methods to the choice of distance function?**
Grouped bar chart comparing Euclidean vs Manhattan for ProtoNet and NCC across all shot counts. Euclidean consistently wins. Includes the average gap across all conditions (~8 pp). Key insight: L2 aligns with the isotropic structure of ResNet-50 feature distributions.

### `rq6_per_class_variability.png`
**RQ6 — Which novel species are hardest to classify and why?**
Per-class accuracy heatmap across all methods and shot counts. Sorted from hardest (left) to easiest (right). ALTAMIRA YELLOWTHROAT (avg 56.1%) and AMERICAN KESTREL (avg 66.2%) are consistently the most challenging. AMERICAN AVOCET and AMERICAN FLAMINGO are consistently the easiest (avg >90%).

### `rq7_stability.png`
**RQ7 — How stable are FSL evaluations across episodes?**
Box plots of per-episode accuracy distributions for each method at 5-shot (B15/N5). Shows spread and outliers across 600 evaluation episodes. ProtoNet has the tightest distribution (CI ±0.10%); MatchingNet has the widest (CI ±0.59%). Confirms that 600 episodes provides stable estimates for all methods.

---

## 8. B10/N10 Split — Dedicated Analysis

### `b10n10_accuracy_bar.png`
Bar chart of mean accuracy for all FSL methods at 1, 3, 5 shot on the B10/N10 split. ProtoNet Euclidean peaks at 95.77%; MatchingNet and RelationNet plateau around 65–67%. Enables direct visual comparison with B15/N5 results.

### `b10n10_kshot_trend.png`
Line plot of accuracy vs K for all methods on B10/N10. Mirrors the B15/N5 trend plots but highlights that with fewer base classes, all methods show lower absolute accuracy and the NCC–ProtoNet gap narrows.

### `b10n10_per_class_heatmap_1shot.png`
Per-class accuracy heatmap for all FSL methods at 1-shot on B10/N10 (10 novel classes). Shows which of the 10 novel species are most/least separable with minimal support.

### `b10n10_per_class_heatmap_3shot.png`
Same heatmap at K=3 shot on B10/N10. Displays accuracy improvements per class from the additional support shots.

### `b10n10_per_class_heatmap_5shot.png`
Same heatmap at K=5 shot on B10/N10. Represents the best achievable per-class accuracy for each method on this more challenging split.

---

## 9. Cross-Method Comparison

### `comparison_accuracy_bar.png`
Grouped bar chart comparing all methods (NCC Euc, ProtoNet Euc, MatchingNet, RelationNet) at all shot counts, pooled across both splits. Primary comparison figure for the thesis results chapter.

### `comparison_bar_splits.png`
Side-by-side bar charts showing B15/N5 (left panel) vs B10/N10 (right panel) for all methods at 5-shot. Highlights the consistent ~4 pp advantage of the richer base split.

### `comparison_delta_splits.png`
Delta plot showing (B15/N5 accuracy − B10/N10 accuracy) per method per shot. Positive values indicate B15/N5 is better. Confirms the B15 advantage is consistent and largest for ProtoNet at 1-shot (+5.6 pp).

### `comparison_kshot_trend_both_splits.png`
Multi-panel line plot: one subplot per method, each showing K-shot trends for both B15/N5 (solid) and B10/N10 (dashed). Shows the parallel structure of shot-count scaling across splits and methods.

### `comparison_macro_f1.png`
Macro F1 score comparison across all methods, shots, and splits. Closely mirrors the accuracy plot (r ≈ 0.99 correlation) but highlights class-balance aspects: MatchingNet has higher F1 variance due to catastrophic failure on individual classes (e.g., AMERICAN KESTREL at 15.9%).

### `per_class_heatmap_fsl.png`
Combined per-class accuracy heatmap across ALL FSL experiments (both splits, all shots, all methods) on novel classes. Largest single-view summary of per-species difficulty. Rows = method/shot/split combinations, columns = novel species sorted by average difficulty.

---

## 10. K-Shot Efficiency

### `kshot_efficiency_curve.png`
Two-panel figure:
- **Left:** Absolute accuracy vs K for all methods on both splits
- **Right:** Marginal accuracy gain per additional shot (diminishing returns curve)

Key finding: The 1→3 shot gain averages ~7 pp across methods, while the 3→5 shot gain averages <1 pp. Practical implication for conservation: 3 field photographs per species is the cost-effective threshold.

---

## 11. Failure Case Analysis

### `failure_case_heatmap.png`
Per-class accuracy heatmap restricted to B15/N5 experiments only, with novel classes sorted from worst to best average accuracy. Clearly shows ALTAMIRA YELLOWTHROAT as the consistently hardest class across all 12 method/shot combinations. Used to support the "visual heterogeneity" argument in the thesis introduction.

### `failure_case_sample_images.png`
Grid of sample images for the 3 hardest novel species (ALTAMIRA YELLOWTHROAT, AMERICAN KESTREL, AMERICAN BITTERN). 4 images per species from the test set. Visually illustrates why these species are harder: subtle plumage patterns, variable lighting, and background clutter confound the classifier.

---

## 12. Problem Case Deep Dives

### `analysis_bittern_inverse_scaling.png`
Special case study for AMERICAN BITTERN, which shows an anomalous inverse K-shot scaling pattern in some experiments (accuracy decreases from 3-shot to 5-shot for certain methods). Plots per-method accuracy vs K for AMERICAN BITTERN specifically, alongside sample images to illustrate the high within-class variance hypothesis.

### `analysis_kestrel_perclass.png`
Per-method accuracy breakdown for AMERICAN KESTREL across all shot counts and both splits. Illustrates the extreme method-dependent variance: ProtoNet achieves 99.0% at 5-shot while MatchingNet achieves only 15.9% — an 83 pp gap on the same species with the same images.

### `analysis_kestrel_samples.png`
Sample images from the AMERICAN KESTREL training and test sets. Displays the visual challenge: male/female dimorphism, varying poses, and colour similarity with base-class raptors. Used to explain why attention-based methods (MatchingNet) fail while prototype-based methods (ProtoNet) succeed.

### `analysis_tsne_kestrel_comparison.png`
t-SNE visualisation of AMERICAN KESTREL embeddings compared against all other novel classes. Shows the ProtoNet-trained feature space where Kestrel embeddings form a well-separated cluster (explaining high ProtoNet accuracy) versus the pre-trained ImageNet space where Kestrel overlaps with other species (explaining MatchingNet failure).

### `analysis_manhattan_negative.png`
Ablation showing why Manhattan distance underperforms Euclidean for this task. Includes distribution plots of pairwise distances (within-class vs between-class) under both metrics, showing that Manhattan distance produces less separable distributions in the 2048-d ResNet-50 feature space.

### `analysis_relationnet_convergence.png`
Training convergence curves for RelationNet across both splits. Shows train/validation loss and accuracy per epoch. Highlights that RelationNet converges slower and to a worse optimum than ProtoNet, consistent with the hypothesis that the MLP relation module is harder to optimise under sparse episodic supervision.
