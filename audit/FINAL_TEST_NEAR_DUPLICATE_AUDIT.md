# Final-Test Cross-Split Near-Duplicate Audit

## Status and provenance

This audit was completed on **2026-09-13**. This document archives the
retained result; it does not report a new or recomputed experiment.

The original script, structured JSON output, and candidate CSV are retained
under [`audit/evidence/final_test_dhash_2026-09-13/`](evidence/final_test_dhash_2026-09-13/):

- [`inspect_data_and_entrypoints.py`](evidence/final_test_dhash_2026-09-13/inspect_data_and_entrypoints.py)
- [`data_entrypoint_evidence.json`](evidence/final_test_dhash_2026-09-13/data_entrypoint_evidence.json)
- [`dhash_candidates.csv`](evidence/final_test_dhash_2026-09-13/dhash_candidates.csv)

The script is preserved as a historical provenance snapshot and retains the
original workspace-relative paths and unrelated entry-point diagnostics. It is
not presented as a portable one-command rerun for a fresh clone.

The existing repository implementation of the 64-bit difference hash is
[`tools/check_split_visual_leakage.py`](../tools/check_split_visual_leakage.py).
The historical audit script imports that implementation rather than defining
a second hash algorithm.

## Scope and retained results

The complete development set contains 4,127 images: 3,002 Train, 524
Validation, and 601 Test-dev images. The frozen Final-Test contains 300
images. The retained audit reports the following cross-set comparisons.

| Comparison | Image pairs | SHA-256 exact duplicates | dHash candidates (Hamming <= 2) | Minimum dHash distance |
|---|---:|---:|---:|---:|
| Train / Final-Test | 900,600 | 0 | 0 | 7 |
| Validation / Final-Test | 157,200 | 0 | 0 | 10 |
| Test-dev / Final-Test | 180,300 | 0 | 0 | 7 |
| Complete Development / Final-Test | 1,238,100 | 0 | 0 | 7 |

The retained candidate CSV contains only its header because no comparison met
the frozen near-duplicate candidate threshold.

## Method

Exact file duplicates were tested through equality of image SHA-256 hashes.
Near-duplicate screening used a 64-bit difference hash (`dHash`):

1. convert each image to grayscale;
2. resize it to 9 x 8 pixels with Lanczos interpolation;
3. compare horizontally adjacent pixel intensities to obtain 64 bits;
4. compute the Hamming distance between the two hashes;
5. flag a candidate when the Hamming distance is at most 2.

All 4,427 images used by this audit decoded successfully at 800 x 450.

## Interpretation and limitation

The retained evidence supports the following bounded statement:

> No SHA-256 exact image duplicate and no 64-bit dHash near-duplicate
> candidate at Hamming distance <= 2 was found between the 300-image frozen
> Final-Test and the complete 4,127-image development set.

This is an image-level leakage screen. It does **not** establish geographic
independence, route independence, collection-date independence, or camera
independence. Difference hashing may also fail to associate views of the same
road segment after substantial changes in viewpoint, crop, illumination, or
time. Those forms of independence require acquisition records and scene-group
provenance beyond this image-hash audit.
