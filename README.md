# Emergency Outer Boundary Detection

Research code for detecting the outer boundary of highway emergency lanes in
maintenance, accident, and other visually complex scenes. The repository
contains the SegFormer-B1 experiments used in the paper, unified boundary
evaluation, external MMSegmentation baselines, and the complete frozen V4.3
reliability-validation and output-gating system.

## Scope

- **A0**: SegFormer-B1 with a fixed 12 px Band target.
- **A1**: A0 with a deep Boundary Head and Thin Edge auxiliary supervision.
- **A2**: A1 with low-weight Soft Heatmap auxiliary supervision.
- **V4.3**: A2 Band-only deployment output + project lane detector + V2
  lane-geometry validation + semantic presence verification + causal
  three-frame median gating.
- **External baselines**: DeepLabV3+ R50 and PIDNet-S through MMSegmentation.

V4.3 is a selective output gate. It does not move or refit the raw boundary.
Only predictions accepted by the geometry and presence checks are returned as
the final `right_boundary` output.

## Repository layout

```text
configs/                 Frozen experiment and system configurations
dataset_manifests/       Immutable split/protocol metadata (no images)
results/                 Paper tables, runtime records, and figures
third_party/ufld/        Minimal Ultra-Fast-Lane-Detection dependency
third_party/mmseg_.../   Custom MMSegmentation dataset/loss/config adapters
tools/                   Conversion, evaluation, export, and V4.3 tools
train_emergency_boundary_baseline.py
```

## Installation

Python 3.10 is recommended. Install PyTorch for the CUDA version available on
your machine, then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

External DeepLabV3+/PIDNet-S experiments additionally require:

```powershell
python -m pip install -r requirements-mmseg.txt
```

The experiments were validated with PyTorch 2.9.1, CUDA 12.8,
MMSegmentation 1.2.2, MMEngine 0.10.7, and MMCV-Lite 2.1.0.

## Data

The public dataset is distributed separately to keep Git history small. Place
the generated development and Final-Test datasets at:

```text
datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601
datasets/emergency_outer_boundary_final_test_fixed12_v1_frozen300
```

The development protocol contains 3,002 training, 524 validation, and 601
Test-dev images. Final-Test contains 300 frozen images and is never merged into
training, validation, or Test-dev.

To regenerate the fixed 12 px Band, 5 px Edge, and Gaussian Heatmap labels from
LabelMe annotations:

```powershell
python tools/convert_labelme_boundary_to_masks.py `
  --root DATASET/train_val DATASET/test `
  --out datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601 `
  --split-dir dataset_manifests/development/splits `
  --fixed-band-width 12 `
  --edge-width 5 `
  --heatmap-radius 9 `
  --heatmap-sigma 3 `
  --copy-images `
  --copy-json
```

Use the frozen CSV files under `dataset_manifests/` to reproduce the official
splits. The raw LabelMe data and frozen protocol files are supplied in the
separate dataset artifact.

## Weights

Download the seven core weights from the GitHub Release and place them under
`weights/`:

```text
a0_band_only.pt
a1_band_edge03.pt
a2_band_edge03_heatmap01.pt
project_lane_detector.pth
v4_presence_verifier.pt
deeplabv3plus_r50.pth
pidnet_s.pth
```

The nine C0/C1/C2 three-seed checkpoints are provided in the Zenodo
supplementary artifact. Optimizer states are removed from all release weights;
their tensors are verified against the original training checkpoints.

The generic SegFormer-B1 pretraining checkpoint is not redistributed. Obtain
it from the upstream SegFormer model zoo and place it at
`weights/pretrained/segformer-B1.pth` before training from scratch.

## Train

```powershell
python train_emergency_boundary_baseline.py `
  --config configs/main/a1_band_edge03.yaml `
  --seed 42 `
  --output-dir runs/a1_seed42 `
  --skip-final-test
```

Replace the config with `a0_band_only.yaml` or
`a2_band_edge03_heatmap01.yaml` for the other main experiments. Checkpoint
selection uses validation IoU; Test-dev is not used for checkpoint selection.

## Evaluate raw output

```powershell
python tools/export_boundary_predictions.py `
  --config configs/main/a2_band_edge03_heatmap01.yaml `
  --checkpoint weights/a2_band_edge03_heatmap01.pt `
  --split test `
  --out-dir outputs/a2_test

python tools/evaluate_raw_boundary_metrics.py `
  --gt-dir datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/masks_band `
  --pred-dir outputs/a2_test/pred_masks `
  --split-csv datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601/splits/test.csv `
  --out-dir outputs/a2_metrics
```

Formal masks use the frozen protocol: resize probability to 800x450, apply
`probability >= 0.5`, and save binary values 0/255. Boundary F1 uses a 5 px
tolerance.

## Run the complete V4.3 system

```powershell
python tools/run_final_v4_3_gate.py `
  --config configs/system/v4_3_final_gate.yaml `
  --split test `
  --out-dir outputs/v4_3_test
```

Frozen V4.3 parameters include `ratio_min=0.30`, `ratio_max=2.0`, ten
bottom-up image-space intervals, presence threshold 0.05, and a causal median
window of three frames.

## Reproducibility

- Official protocol and checkpoint hashes are shipped with the release assets.
- Controlled C0/C1/C2 experiments use seeds 42, 2026, and 3407.
- Reported metrics, Final-Test results, runtime measurements, and qualitative
  figures are under `results/`.
- Final-Test was evaluated only after models and gate thresholds were frozen.
- The retained [Final-Test cross-split near-duplicate audit](audit/FINAL_TEST_NEAR_DUPLICATE_AUDIT.md)
  found no SHA-256 exact duplicate or 64-bit dHash candidate at Hamming
  distance `<= 2` between Final-Test and the complete development set.

## License and attribution

Project-authored code is released under the MIT License. The vendored minimal
UFLD code remains under its MIT license, while the custom MMSegmentation
adapter depends on Apache-2.0 MMSegmentation. See `THIRD_PARTY_NOTICES.md`.
