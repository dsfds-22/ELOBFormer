"""Build the immutable report bundle for the frozen 300-frame Final-Test."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "final_test_frozen300"
RAW_ROOT = REPORT_ROOT / "unified_metrics"
GATE_ROOT = REPORT_ROOT / "v4_3_unified_metrics"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def value(row, field, percent=False, digits=4):
    raw = row.get(field, "")
    if raw in (None, ""):
        return "-"
    number = float(raw)
    if percent:
        return f"{number * 100:.2f}%"
    return f"{number:.{digits}f}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def create_prediction_manifest():
    sources = {
        "A0": REPORT_ROOT / "predictions" / "a0" / "pred_masks",
        "A1": REPORT_ROOT / "predictions" / "a1" / "pred_masks",
        "A2": REPORT_ROOT / "predictions" / "a2" / "pred_masks",
        "DeepLabV3+": REPORT_ROOT / "predictions" / "deeplabv3plus_r50" / "pred_masks",
        "PIDNet-S": REPORT_ROOT / "predictions" / "pidnet_s" / "pred_masks",
        "V4.3": REPORT_ROOT / "v4_3_gate" / "gated_masks_model",
    }
    rows = []
    for model, directory in sources.items():
        files = sorted(directory.glob("*.png"))
        if len(files) != 300:
            raise RuntimeError(f"{model}: expected 300 masks, found {len(files)}")
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((model, path.relative_to(ROOT).as_posix(), digest))
    manifest = REPORT_ROOT / "prediction_masks_sha256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "path", "sha256"))
        writer.writerows(rows)


def main():
    frames = read_csv(RAW_ROOT / "frame_macro_summary.csv")
    scenes = read_csv(RAW_ROOT / "per_scene_metrics.csv")
    scene_macro = {
        row["model"]: row
        for row in read_csv(RAW_ROOT / "scene_macro_summary.csv")
        if row["group"] == "overall"
    }
    subsets = read_csv(RAW_ROOT / "subset_frame_summary.csv")
    gate_frame = read_csv(GATE_ROOT / "frame_macro_summary.csv")[0]
    gate_scenes = read_csv(GATE_ROOT / "per_scene_metrics.csv")
    gate_subsets = read_csv(GATE_ROOT / "subset_frame_summary.csv")
    gate_system = json.loads(
        (REPORT_ROOT / "v4_3_gate" / "system_gate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    all_scenes = scenes + gate_scenes
    all_subsets = subsets + gate_subsets
    create_prediction_manifest()

    lines = [
        "# Frozen Final-Test Evaluation Report",
        "",
        "## Protocol",
        "",
        "- Status: frozen one-time evaluation; Final-Test was not used for training, checkpoint selection, threshold selection, or post-result tuning.",
        "- Source: public archive directory `final_test_labelme`.",
        "- Isolated dataset: `datasets/emergency_outer_boundary_final_test_fixed12_v1_frozen300`.",
        "- Composition: 300 frames, 13 scene groups, 208 positive frames, 46 hard-negative frames, and 46 OOD-negative frames.",
        "- Isolation check: Final-Test prefixes 050-062 do not occur in the development splits, and SHA-256 comparison found zero exact image duplicates against the 4,127 development images.",
        "- Split containment: the isolated dataset contains 0 training rows, 0 validation rows, and 300 test rows.",
        "- Labels: human-reviewed LabelMe polylines rasterized with OpenCV `LINE_8` and nominal thickness 12 px; no adaptive-width band was used.",
        "- Output protocol: probability >= 0.5, exported and evaluated as binary 0/255 masks at 800x450.",
        "- Boundary matching tolerance: 5 px. Empty positive predictions receive the 800x450 diagonal penalty, 917.877988 px.",
        "- V4.3 is frozen at `ratio_min=0.30`, `ratio_max=2.0`, boundary threshold 0.5, presence threshold 0.05, and causal median window 3.",
        "",
        "## Frame-Level Raw Outputs",
        "",
    ]
    lines += markdown_table(
        ["Model", "All IoU", "Pos IoU", "Pos Dice", "Empty", "Neg FPR", "BF1", "PMBE px", "CMBE px", "Coverage", "Continuity"],
        [
            [
                row["model"], value(row, "all_iou"), value(row, "positive_iou"),
                value(row, "positive_dice"), value(row, "positive_empty_rate", True),
                value(row, "negative_fpr", True), value(row, "positive_bf1"),
                value(row, "pmbe", digits=2), value(row, "cmbe", digits=2),
                value(row, "coverage"), value(row, "continuity"),
            ]
            for row in frames
        ],
    )
    lines += [
        "",
        "A1 is the strongest raw positive-boundary detector: it has the highest Positive IoU, Dice, BF1, Coverage, and Continuity, and the lowest PMBE. Its main weakness is existence discrimination: it predicts a non-empty boundary in 94.57% of negative frames. DeepLabV3+ has the highest All IoU because it rejects substantially more negatives, while its positive-boundary geometry is weaker than A1.",
        "",
        "## Thirteen-Scene Macro Average",
        "",
    ]
    lines += markdown_table(
        ["Model", "All IoU", "Pos IoU", "Empty", "Neg FPR", "BF1", "PMBE px", "CMBE px", "Coverage", "Continuity"],
        [
            [
                model, value(row, "all_iou"), value(row, "positive_iou"),
                value(row, "positive_empty_rate", True), value(row, "negative_fpr", True),
                value(row, "positive_bf1"), value(row, "pmbe", digits=2),
                value(row, "cmbe", digits=2), value(row, "coverage"),
                value(row, "continuity"),
            ]
            for model, row in scene_macro.items()
        ],
    )
    lines += [
        "",
        "Scene macro averaging gives every scene group equal weight. It closely agrees with the frame-level ranking because the 13 groups are nearly balanced (23 frames each, except group 057 with 24).",
        "",
        "## Positive Scene Groups",
        "",
    ]
    positive_models = ["A0", "A1", "A2", "DeepLabV3+", "PIDNet-S", "V4.3"]
    positive_scene_rows = []
    for scene in [f"final_{index:03d}" for index in (52, 53, 54, 55, 56, 57, 58, 60, 61)]:
        row = [scene]
        for model in positive_models:
            found = next(item for item in all_scenes if item["model"] == model and item["group"] == scene)
            row.append(value(found, "positive_bf1"))
        positive_scene_rows.append(row)
    lines += markdown_table(["Scene", *[f"{model} BF1" for model in positive_models]], positive_scene_rows)
    lines += ["", "## Negative Scene Groups", ""]
    negative_scene_rows = []
    for scene in [f"final_{index:03d}" for index in (50, 51, 59, 62)]:
        row = [scene]
        subset = next(item["subset"] for item in all_scenes if item["group"] == scene)
        row.append(subset)
        for model in positive_models:
            found = next(item for item in all_scenes if item["model"] == model and item["group"] == scene)
            row.append(value(found, "negative_fpr", True))
        negative_scene_rows.append(row)
    lines += markdown_table(["Scene", "Subset", *[f"{model} FPR" for model in positive_models]], negative_scene_rows)
    lines += ["", "## Subset Results", ""]
    subset_rows = []
    for model in positive_models:
        for subset in ("positive", "hard_negative", "ood_negative"):
            row = next(item for item in all_subsets if item["model"] == model and item["group"] == subset)
            subset_rows.append([
                model, subset, row["samples"], value(row, "positive_iou"),
                value(row, "positive_bf1"), value(row, "positive_empty_rate", True),
                value(row, "negative_fpr", True), value(row, "pmbe", digits=2),
                value(row, "cmbe", digits=2),
            ])
    lines += markdown_table(
        ["Model", "Subset", "N", "Pos IoU", "BF1", "Empty", "FPR", "PMBE px", "CMBE px"],
        subset_rows,
    )
    lines += [
        "",
        "## Frozen V4.3 Output Gate",
        "",
        f"- Positive acceptance rate: {gate_system['positive_acceptance_rate'] * 100:.2f}% ({gate_system['accepted_positive_samples']}/{gate_system['positive_samples']}).",
        f"- Negative rejection rate: {gate_system['negative_rejection_rate'] * 100:.2f}% ({gate_system['correctly_rejected_negative_samples']}/{gate_system['negative_samples']}).",
        f"- Post-gate FPR: {gate_system['post_gate_false_positive_rate'] * 100:.2f}% (frame-level Wilson 95% CI {gate_system['frame_level_fpr_wilson95_low'] * 100:.2f}% to {gate_system['frame_level_fpr_wilson95_high'] * 100:.2f}%).",
        f"- Accepted-boundary BF1/CMBE: {gate_system['accepted_boundary_f1']:.4f} / {gate_system['accepted_mean_boundary_error_px']:.2f} px.",
        f"- End-to-end positive BF1/PMBE: {gate_frame['positive_bf1']} / {float(gate_frame['pmbe']):.2f} px.",
        f"- V2 geometry-state unavailable: {gate_system['v2_geometry_unavailable_ratio'] * 100:.2f}% ({gate_system['v2_geometry_unavailable_samples']}/300).",
        f"- Final total empty-output ratio: {gate_system['final_empty_output_ratio'] * 100:.2f}% ({gate_system['final_empty_output_samples']}/300), containing both correct negative rejection and rejected positives.",
        f"- Full pipeline: {gate_system['full_pipeline_fps']:.2f} FPS, {gate_system['full_pipeline_mean_ms']:.2f} ms/frame.",
        "",
        "V4.3 is a selective output gate. It does not improve the raw A2 segmentation mask; it suppresses unreliable outputs. Its accepted-sample CMBE therefore must not be compared as though it were an unconditional segmentation metric. The end-to-end PMBE includes the 14 rejected positive frames and is correspondingly high.",
        "",
        "## Interpretation",
        "",
        "1. A1 generalizes best for raw positive-boundary detection on this genuinely unseen set, confirming the value of the thin-edge auxiliary supervision.",
        "2. A2 does not reproduce its Test-dev advantage on Final-Test. Heatmap supervision reduces A1's OOD-negative FPR but lowers positive geometry quality and still accepts every hard-negative frame before gating.",
        "3. The raw models' hardest issue is the hard-negative subset, where road edges or barriers resemble an emergency outer boundary. A1 and A2 both reach 100% FPR on these 46 frames.",
        "4. V4.3 directly addresses that system-level weakness: hard-negative FPR falls to 2.17% and OOD-negative FPR to 0%, at the cost of rejecting 6.73% of positive frames.",
        "5. No model, threshold, or gate parameter should be changed on the basis of these results. Any future method change requires a new development experiment and a separately collected untouched test set.",
        "",
        "## Artifacts",
        "",
        "- Frozen protocol: `reports/final_test_frozen300/protocol.yaml`",
        "- Frame metrics: `reports/final_test_frozen300/unified_metrics/per_frame_metrics.csv`",
        "- Scene metrics: `reports/final_test_frozen300/unified_metrics/per_scene_metrics.csv`",
        "- Scene macro: `reports/final_test_frozen300/unified_metrics/scene_macro_summary.csv`",
        "- Subset metrics: `reports/final_test_frozen300/unified_metrics/subset_frame_summary.csv`",
        "- V4.3 system metrics: `reports/final_test_frozen300/v4_3_gate/system_gate_metrics.json`",
        "- Frozen input hashes: `reports/final_test_frozen300/frozen_artifacts_sha256.csv`",
        "- Prediction mask hashes: `reports/final_test_frozen300/prediction_masks_sha256.csv`",
        "- Integrity QA: all 1,800 raw/gated masks are 800x450 binary 0/255 images; V4.3's internal raw Band-only masks match the frozen A2 masks in all 300 cases by SHA-256.",
    ]
    (REPORT_ROOT / "final_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_ROOT / "final_test_report.md")


if __name__ == "__main__":
    main()
