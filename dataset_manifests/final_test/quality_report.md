# Boundary Training Label Quality Report

- Samples: 300
- Positive: 208
- Negative: 92
- Expected size: 800x450
- Issues: 0

| Split | Samples | Positive | Negative |
|---|---:|---:|---:|
| train | 0 | 0 | 0 |
| val | 0 | 0 | 0 |
| test | 300 | 208 | 92 |

## Label Checks

- Band masks are binary 0/255 images.
- Edge masks are binary 0/255 images.
- Every edge mask is contained inside its band mask.
- Positive samples have non-empty band, edge, and heatmap labels.
- Negative samples have empty band, edge, and heatmap labels.
- Every converted sample appears in exactly one split.
