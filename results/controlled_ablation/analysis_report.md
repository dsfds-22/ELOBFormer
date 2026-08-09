# Controlled Ablation Analysis

## Protocol

- Dataset: `emergency_outer_boundary_grouped_fixed12_v2_clean601`
- Split counts: Train 3002, Val 524, Test-dev 601
- Seeds: 42, 2026, 3407
- Selection: best validation IoU with patience 15
- Formal prediction: Band probability >= 0.5, exported as binary 0/255 mask
- Boundary tolerance: 5 px
- Final-Test was not used

## Training and Validation

| Group | Best Val IoU (mean +/- SD) | Best Val Dice (mean +/- SD) | Epochs run |
|---|---:|---:|---|
| C0 Shallow Band | 0.5074 +/- 0.0146 | 0.5775 +/- 0.0173 | 42 / 32 / 63 |
| C1 Deep Band, Edge weight 0 | **0.5208 +/- 0.0181** | **0.5927 +/- 0.0202** | 47 / 30 / 26 |
| C2 Deep Band, Edge weight 0.3 | 0.4828 +/- 0.0290 | 0.5613 +/- 0.0380 | 26 / 32 / 25 |

All nine runs completed without non-finite losses. Every run stopped exactly 15
epochs after its best validation IoU. C1 improves validation IoU over C0 for all
three matched seeds. C2 validation IoU is lower than C1 for all three seeds.

## Unified Test-dev Results

Values are mean +/- sample SD over the three seeds.

| Group | All IoU | Positive IoU | Positive BF1 | PMBE (px) | CMBE (px) | Empty Rate | Negative FPR | Coverage | Continuity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.6134 +/- 0.0076 | 0.6735 +/- 0.0048 | 0.8578 +/- 0.0091 | 46.47 +/- 13.57 | 6.61 +/- 0.07 | 4.37% +/- 1.49% | 92.78% +/- 8.55% | 0.8850 +/- 0.0150 | 0.8639 +/- 0.0095 |
| C1 | 0.6165 +/- 0.0006 | 0.6744 +/- 0.0096 | 0.8602 +/- 0.0137 | 26.27 +/- 6.69 | 8.33 +/- 2.94 | 1.97% +/- 0.87% | 90.56% +/- 9.18% | 0.8871 +/- 0.0125 | 0.8702 +/- 0.0078 |
| C2 | **0.6336 +/- 0.0131** | **0.6909 +/- 0.0046** | **0.8811 +/- 0.0116** | **16.00 +/- 10.71** | **4.76 +/- 1.14** | **1.23% +/- 1.08%** | **88.33% +/- 12.02%** | **0.9127 +/- 0.0190** | **0.8981 +/- 0.0161** |

## Controlled Comparisons

### C1 minus C0: deep Band Head

- Mean BF1 change: +0.00244.
- Per-seed BF1 changes: -0.01113, -0.00410, +0.02256.
- Scene-cluster bootstrap 95% CI: [-0.01926, +0.02400].
- Mean PMBE change: -20.20 px, but its 95% CI includes zero.

The deeper Band Head alone is not supported as a stable BF1 improvement on
Test-dev. It reduces complete misses on average, but the direction and magnitude
of geometric improvement vary across seeds and scenes. H1 is therefore not
confirmed by the primary metric.

### C2 minus C1: Thin Edge auxiliary supervision

- Mean BF1 change: +0.02088.
- Per-seed BF1 changes: +0.04828, +0.01607, -0.00172.
- Scene-cluster bootstrap 95% CI: [-0.00560, +0.06265].
- Mean PMBE change: -10.27 px; 95% CI [-25.82, -0.59].
- Mean CMBE changes by seed: -2.11, -6.93, -1.68 px.
- Mean Positive IoU change: +0.01649.

Thin Edge has a meaningful average benefit and lowers PMBE across scene-cluster
bootstrap samples. It also lowers CMBE for all three matched seeds. However, BF1
does not improve for all three seeds and its confidence interval includes zero.
Under the preregistered decision rule this is **limited support**, not strong
support, for H2.

### C2 minus C0: complete method

- Mean BF1 change: +0.02332.
- Per-seed BF1 changes: +0.03715, +0.01197, +0.02084; all three are positive.
- Scene-cluster bootstrap BF1 95% CI: [-0.00415, +0.07408].
- Mean PMBE change: -30.47 px; 95% CI [-88.89, -2.03].
- Positive IoU increases for all three matched seeds.

The complete Deep Band plus Thin Edge design is consistently better than the
shallow baseline by per-seed BF1 and Positive IoU direction. Its BF1 cluster CI
still includes zero because only nine Test-dev scene groups contain positive
samples and scene difficulty is highly heterogeneous.

## Interpretation

1. Validation and Test-dev rankings disagree: C1 is best on Val, while C2 is
   best on Test-dev. This indicates a distribution difference and confirms that
   validation IoU alone cannot describe boundary generalization.
2. The isolated deep Band Head contribution is weak. Earlier A0-to-A1 gains
   cannot be attributed solely to the larger output head.
3. Thin Edge supervision is the more valuable component in the controlled
   experiment. Its clearest evidence is lower PMBE, lower CMBE across all seeds,
   higher mean BF1, and improved coverage/continuity.
4. The preregistered strong-support criterion is not met because C2-to-C1 BF1 is
   slightly negative for seed 3407 and the scene-cluster BF1 CI crosses zero.
5. Negative FPR remains high for all raw segmentation models. This experiment
   evaluates raw boundary learning and does not replace the frozen V4.3 output
   gating module.

## Formal Conclusion

The experiment provides limited but substantive support for Thin Edge auxiliary
supervision. The complete C2 model gives the strongest three-seed Test-dev mean
performance, while the deep Band Head alone does not provide stable primary-metric
improvement. The paper should describe Thin Edge as improving average boundary
quality and miss-sensitive error, while explicitly reporting its residual
seed/scene variability.
