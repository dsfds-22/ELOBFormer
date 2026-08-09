# Chapter 4 Qualitative Source Images (clean601)

This directory contains only the five original RGB images selected for the
qualitative analysis in Chapter 4. No image has been resized, recompressed, or
pre-composed. All samples belong to the formal `fixed12_v2_clean601` Test-dev
split.

## Selected cases

| Order | Sample | Intended qualitative analysis | Selection evidence |
|---:|---|---|---|
| 1 | `034_0043_rgb` | Clear A0 to A1 improvement | A0 is empty (BF1 0.0000); A1 recovers the boundary (BF1 0.9374, PMBE 1.54 px); A2 BF1 0.7742 |
| 2 | `028_0218_rgb` | A2 recovers an A1 empty prediction | A1 is empty; A2 produces a valid partial boundary (BF1 0.5728, PMBE 29.70 px). This illustrates recovery rather than perfect localization |
| 3 | `033_0002_rgb` | Near-end under-extension failure | A1/A2 both detect the far/middle boundary but under-cover the near end; A2 Coverage is 0.4381 and PMBE is 31.62 px |
| 4 | `049_00008` | Negative false positive rejected by V4.3 | The actual Band-only Deploy input to V4.3 contains 4193 false-positive pixels; V4.3 marks it `low_reliability` and produces an empty final output. The A2 Band output is retained only as supporting evidence, not as a substitute for the strict before/after path |
| 5 | `047_00023` | Five-model horizontal comparison | BF1: A0 0.8726, A1 0.9970, A2 0.9988, DeepLabV3+ 1.0000, PIDNet-S 0.2941 |

## Formal artifact locations

For each sample ID `<id>`:

- Original RGB: `datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/images/<id>.jpg`
- GT Band: `datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/masks_band/<id>.png`
- A0 mask: `reports/formal_predictions_v2_clean601/a0_band/pred_masks/<id>.png`
- A1 mask: `reports/formal_predictions_v2_clean601/a1_edge03/pred_masks/<id>.png`
- A2 mask: `reports/formal_predictions_v2_clean601/a2_heatmap01/pred_masks/<id>.png`
- DeepLabV3+ mask: `runs/emergency_boundary/deeplabv3plus_r50_band_grouped_fixed12_sqrt_scene_v1_80epoch_es15/full_test_best_epoch52/pred_masks/<id>.png`
- PIDNet-S mask: `runs/emergency_boundary/pidnet_s_band_edge_grouped_fixed12_sqrt_scene_v1_80epoch_es15/full_test_best_epoch19/pred_masks/<id>.png`
- V4.3 actual Band-only Deploy input: `reports/v4_3_band_only_deploy_v2_clean601/raw_boundary_masks/<id>.png`
- V4.3 gated mask: `reports/v4_3_band_only_deploy_v2_clean601/gated_masks_model/<id>.png`
- V4.3 decision fields: `reports/v4_3_band_only_deploy_v2_clean601/image_summary.csv`

All formal prediction masks above are binary `0/255` PNG files. The negative
sample has an empty GT Band by definition. For the fourth figure, the strict
comparison is Original RGB -> empty GT -> V4.3 actual Band-only Deploy input ->
V4.3 gated output. The separately evaluated A2 mask may be shown only as a
reference. Quantitative values are from
`reports/unified_mask_evaluation_v2_clean601/per_image_metrics.csv` and the V4.3
decision/output files listed above.

## Prepared figure groups

The requested source files and paper-ready comparison figures are in
`qualitative_groups/<sample_id>/`. Each group contains exact copies of the
required RGB/mask artifacts, a binary-mask comparison, and an RGB-overlay
comparison. Both comparisons are available as PNG previews and lossless LZW
TIFF files at 600 DPI.
