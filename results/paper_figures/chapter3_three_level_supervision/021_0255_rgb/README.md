# Three-level Supervision Example

- Sample: `021_0255_rgb`
- Split: `train`
- Resolution: `800 x 450`
- Boundary annotations: `1` polyline(s), `8` control points
- Band label: fixed total width `12 px`
- Thin-edge label: fixed total width `5 px`
- Soft heatmap: truncated Gaussian, radius `9 px`, sigma `3 px`
- Paper figure: `4048 x 514 px`, lossless LZW TIFF, `600 DPI`

## Source files

- RGB: `datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/images/021_0255_rgb.jpg`
- LabelMe JSON: `development_labelme/train_val/021_0255_rgb.json` in the public dataset archive
- Band: `datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/masks_band/021_0255_rgb.png`
- Edge: `datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/masks_edge/021_0255_rgb.png`
- Heatmap: `datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/heatmaps/021_0255_rgb.png`

The copied grayscale Band, Edge, and Heatmap files are the exact training labels.
`02_labelme_polyline_overlay.png` and `05_heatmap_color.png` are visualization-only files.
`three_level_supervision_figure.png` is a convenient preview.
`three_level_supervision_figure_600dpi.tif` is the paper-ready five-panel figure.
