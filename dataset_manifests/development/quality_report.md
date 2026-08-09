# Boundary Training Label Quality Report

- Samples: 4127
- Positive: 3260
- Negative: 867
- Expected size: 800x450
- Issues: 0

| Split | Samples | Positive | Negative |
|---|---:|---:|---:|
| train | 3002 | 2376 | 626 |
| val | 524 | 343 | 181 |
| test | 601 | 541 | 60 |

## Label Checks

- Band masks are binary 0/255 images.
- Edge masks are binary 0/255 images.
- Every edge mask is contained inside its band mask.
- Positive samples have non-empty band, edge, and heatmap labels.
- Negative samples have empty band, edge, and heatmap labels.
- Every converted sample appears in exactly one split.
