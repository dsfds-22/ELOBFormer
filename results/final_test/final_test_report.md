# Frozen Final-Test Evaluation Report

## Protocol

- Status: frozen one-time evaluation; Final-Test was not used for training, checkpoint selection, threshold selection, or post-result tuning.
- Source: public archive directory `final_test_labelme`.
- Isolated dataset: `datasets/emergency_outer_boundary_final_test_fixed12_v1_frozen300`.
- Composition: 300 frames, 13 scene groups, 208 positive frames, 46 hard-negative frames, and 46 OOD-negative frames.
- Isolation check: Final-Test prefixes 050-062 do not occur in the development splits, and SHA-256 comparison found zero exact image duplicates against the 4,127 development images.
- Split containment: the isolated dataset contains 0 training rows, 0 validation rows, and 300 test rows.
- Labels: human-reviewed LabelMe polylines rasterized with OpenCV `LINE_8` and nominal thickness 12 px; no adaptive-width band was used.
- Output protocol: probability >= 0.5, exported and evaluated as binary 0/255 masks at 800x450.
- Boundary matching tolerance: 5 px. Empty positive predictions receive the 800x450 diagonal penalty, 917.877988 px.
- V4.3 is frozen at `ratio_min=0.30`, `ratio_max=2.0`, boundary threshold 0.5, presence threshold 0.05, and causal median window 3.

## Frame-Level Raw Outputs

| Model | All IoU | Pos IoU | Pos Dice | Empty | Neg FPR | BF1 | PMBE px | CMBE px | Coverage | Continuity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.6205 | 0.7748 | 0.8608 | 0.96% | 72.83% | 0.9413 | 10.49 | 1.68 | 0.9708 | 0.9512 |
| A1 | 0.6081 | 0.8530 | 0.9100 | 0.48% | 94.57% | 0.9572 | 6.42 | 2.01 | 0.9795 | 0.9557 |
| A2 | 0.5818 | 0.7286 | 0.8157 | 0.96% | 75.00% | 0.8995 | 12.84 | 4.06 | 0.9366 | 0.9057 |
| DeepLabV3+ | 0.6771 | 0.7170 | 0.8117 | 1.44% | 41.30% | 0.8941 | 17.08 | 3.90 | 0.9340 | 0.9069 |
| PIDNet-S | 0.6389 | 0.6715 | 0.7693 | 4.81% | 43.48% | 0.8638 | 48.12 | 4.20 | 0.8842 | 0.8659 |

A1 is the strongest raw positive-boundary detector: it has the highest Positive IoU, Dice, BF1, Coverage, and Continuity, and the lowest PMBE. Its main weakness is existence discrimination: it predicts a non-empty boundary in 94.57% of negative frames. DeepLabV3+ has the highest All IoU because it rejects substantially more negatives, while its positive-boundary geometry is weaker than A1.

## Thirteen-Scene Macro Average

| Model | All IoU | Pos IoU | Empty | Neg FPR | BF1 | PMBE px | CMBE px | Coverage | Continuity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.6205 | 0.7755 | 0.93% | 72.83% | 0.9418 | 10.16 | 1.68 | 0.9712 | 0.9518 |
| A1 | 0.6079 | 0.8539 | 0.48% | 94.57% | 0.9577 | 6.40 | 1.97 | 0.9799 | 0.9565 |
| A2 | 0.5817 | 0.7292 | 0.93% | 75.00% | 0.8999 | 12.52 | 4.05 | 0.9369 | 0.9062 |
| DeepLabV3+ | 0.6777 | 0.7180 | 1.39% | 41.30% | 0.8950 | 16.59 | 3.90 | 0.9346 | 0.9078 |
| PIDNet-S | 0.6393 | 0.6722 | 4.77% | 43.48% | 0.8644 | 47.78 | 4.41 | 0.8847 | 0.8666 |

Scene macro averaging gives every scene group equal weight. It closely agrees with the frame-level ranking because the 13 groups are nearly balanced (23 frames each, except group 057 with 24).

## Positive Scene Groups

| Scene | A0 BF1 | A1 BF1 | A2 BF1 | DeepLabV3+ BF1 | PIDNet-S BF1 | V4.3 BF1 |
| --- | --- | --- | --- | --- | --- | --- |
| final_052 | 0.9335 | 0.9718 | 0.8251 | 0.9494 | 0.8669 | 0.8251 |
| final_053 | 0.9590 | 0.9792 | 0.9718 | 0.8779 | 0.6153 | 0.9718 |
| final_054 | 0.9733 | 0.9496 | 0.7984 | 0.9744 | 0.9847 | 0.6665 |
| final_055 | 0.9271 | 0.9563 | 0.8848 | 0.9227 | 0.9438 | 0.8848 |
| final_056 | 0.9926 | 0.9969 | 0.9877 | 0.9938 | 0.9840 | 0.9877 |
| final_057 | 0.8391 | 0.8443 | 0.8207 | 0.7214 | 0.7304 | 0.7385 |
| final_058 | 0.8844 | 0.9745 | 0.8368 | 0.8326 | 0.8593 | 0.8368 |
| final_060 | 0.9848 | 0.9744 | 0.9752 | 0.8054 | 0.8008 | 0.8497 |
| final_061 | 0.9822 | 0.9723 | 0.9980 | 0.9772 | 0.9946 | 0.9980 |

## Negative Scene Groups

| Scene | Subset | A0 FPR | A1 FPR | A2 FPR | DeepLabV3+ FPR | PIDNet-S FPR | V4.3 FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_050 | ood_negative | 56.52% | 91.30% | 52.17% | 21.74% | 4.35% | 0.00% |
| final_051 | ood_negative | 47.83% | 86.96% | 47.83% | 21.74% | 34.78% | 0.00% |
| final_059 | hard_negative | 91.30% | 100.00% | 100.00% | 95.65% | 95.65% | 4.35% |
| final_062 | hard_negative | 95.65% | 100.00% | 100.00% | 26.09% | 39.13% | 0.00% |

## Subset Results

| Model | Subset | N | Pos IoU | BF1 | Empty | FPR | PMBE px | CMBE px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | positive | 208 | 0.7748 | 0.9413 | 0.96% | - | 10.49 | 1.68 |
| A0 | hard_negative | 46 | - | - | - | 93.48% | - | - |
| A0 | ood_negative | 46 | - | - | - | 52.17% | - | - |
| A1 | positive | 208 | 0.8530 | 0.9572 | 0.48% | - | 6.42 | 2.01 |
| A1 | hard_negative | 46 | - | - | - | 100.00% | - | - |
| A1 | ood_negative | 46 | - | - | - | 89.13% | - | - |
| A2 | positive | 208 | 0.7286 | 0.8995 | 0.96% | - | 12.84 | 4.06 |
| A2 | hard_negative | 46 | - | - | - | 100.00% | - | - |
| A2 | ood_negative | 46 | - | - | - | 50.00% | - | - |
| DeepLabV3+ | positive | 208 | 0.7170 | 0.8941 | 1.44% | - | 17.08 | 3.90 |
| DeepLabV3+ | hard_negative | 46 | - | - | - | 60.87% | - | - |
| DeepLabV3+ | ood_negative | 46 | - | - | - | 21.74% | - | - |
| PIDNet-S | positive | 208 | 0.6715 | 0.8638 | 4.81% | - | 48.12 | 4.20 |
| PIDNet-S | hard_negative | 46 | - | - | - | 67.39% | - | - |
| PIDNet-S | ood_negative | 46 | - | - | - | 19.57% | - | - |
| V4.3 | positive | 208 | 0.7007 | 0.8615 | 6.73% | - | 63.82 | 2.18 |
| V4.3 | hard_negative | 46 | - | - | - | 2.17% | - | - |
| V4.3 | ood_negative | 46 | - | - | - | 0.00% | - | - |

## Frozen V4.3 Output Gate

- Positive acceptance rate: 93.27% (194/208).
- Negative rejection rate: 98.91% (91/92).
- Post-gate FPR: 1.09% (frame-level Wilson 95% CI 0.19% to 5.90%).
- Accepted-boundary BF1/CMBE: 0.9237 / 2.18 px.
- End-to-end positive BF1/PMBE: 0.861534 / 63.82 px.
- V2 geometry-state unavailable: 9.33% (28/300).
- Final total empty-output ratio: 35.00% (105/300), containing both correct negative rejection and rejected positives.
- Full pipeline: 26.79 FPS, 37.33 ms/frame.

V4.3 is a selective output gate. It does not improve the raw A2 segmentation mask; it suppresses unreliable outputs. Its accepted-sample CMBE therefore must not be compared as though it were an unconditional segmentation metric. The end-to-end PMBE includes the 14 rejected positive frames and is correspondingly high.

## Interpretation

1. A1 generalizes best for raw positive-boundary detection on this genuinely unseen set, confirming the value of the thin-edge auxiliary supervision.
2. A2 does not reproduce its Test-dev advantage on Final-Test. Heatmap supervision reduces A1's OOD-negative FPR but lowers positive geometry quality and still accepts every hard-negative frame before gating.
3. The raw models' hardest issue is the hard-negative subset, where road edges or barriers resemble an emergency outer boundary. A1 and A2 both reach 100% FPR on these 46 frames.
4. V4.3 directly addresses that system-level weakness: hard-negative FPR falls to 2.17% and OOD-negative FPR to 0%, at the cost of rejecting 6.73% of positive frames.
5. No model, threshold, or gate parameter should be changed on the basis of these results. Any future method change requires a new development experiment and a separately collected untouched test set.

## Artifacts

- Frozen protocol: `reports/final_test_frozen300/protocol.yaml`
- Frame metrics: `reports/final_test_frozen300/unified_metrics/per_frame_metrics.csv`
- Scene metrics: `reports/final_test_frozen300/unified_metrics/per_scene_metrics.csv`
- Scene macro: `reports/final_test_frozen300/unified_metrics/scene_macro_summary.csv`
- Subset metrics: `reports/final_test_frozen300/unified_metrics/subset_frame_summary.csv`
- V4.3 system metrics: `reports/final_test_frozen300/v4_3_gate/system_gate_metrics.json`
- Frozen input hashes: `reports/final_test_frozen300/frozen_artifacts_sha256.csv`
- Prediction mask hashes: `reports/final_test_frozen300/prediction_masks_sha256.csv`
- Integrity QA: all 1,800 raw/gated masks are 800x450 binary 0/255 images; V4.3's internal raw Band-only masks match the frozen A2 masks in all 300 cases by SHA-256.
