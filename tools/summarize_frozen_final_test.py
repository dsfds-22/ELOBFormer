"""Summarize frozen Final-Test predictions without changing model outputs.

The script evaluates already-exported binary masks. It reports frame-level
macros, scene-level macros, and the predefined Final-Test subsets. Positive
boundary errors are split into penalized (PMBE/PMaxBE) and conditional
(CMBE/CMaxBE) variants.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


DIAGONAL_800X450 = math.hypot(800, 450)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="ModelName=prediction_mask_directory; repeat for each model.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=5.0)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_prediction(directory, sample_id):
    candidates = (
        directory / f"{sample_id}.png",
        directory / f"{sample_id}_prob.png",
        directory / f"{sample_id}_mask.png",
        directory / f"{sample_id}_pred.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob(f"{sample_id}*.png"))
    return matches[0] if matches else None


def load_mask(path, threshold):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask >= threshold).astype(np.uint8)


def align_prediction(prediction, target):
    if prediction.shape == target.shape:
        return prediction
    height, width = target.shape
    pred_height, pred_width = prediction.shape
    if pred_height >= height and pred_width >= width:
        return prediction[:height, :width]
    return cv2.resize(
        prediction, (width, height), interpolation=cv2.INTER_NEAREST
    )


def mask_boundary(mask):
    if not mask.any():
        return np.zeros_like(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return ((mask - eroded) > 0).astype(np.uint8)


def distance_to_boundary(source, target):
    if not source.any():
        return np.empty(0, dtype=np.float32)
    if not target.any():
        return np.full(int(source.sum()), DIAGONAL_800X450, dtype=np.float32)
    inverse = (1 - target) * 255
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    return distance[source.astype(bool)].astype(np.float32)


def boundary_metrics(prediction, target, tolerance):
    pred_boundary = mask_boundary(prediction)
    gt_boundary = mask_boundary(target)
    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0, 0.0, 0.0
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0, DIAGONAL_800X450, DIAGONAL_800X450
    pred_to_gt = distance_to_boundary(pred_boundary, gt_boundary)
    gt_to_pred = distance_to_boundary(gt_boundary, pred_boundary)
    precision = float(np.mean(pred_to_gt <= tolerance))
    recall = float(np.mean(gt_to_pred <= tolerance))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    distances = np.concatenate((pred_to_gt, gt_to_pred))
    return float(f1), float(distances.mean()), float(distances.max())


def continuity_metrics(prediction, target, tolerance):
    pred_boundary = mask_boundary(prediction)
    gt_boundary = mask_boundary(target)
    gt_rows = np.flatnonzero(gt_boundary.any(axis=1))
    if not len(gt_rows):
        value = 1.0 if not pred_boundary.any() else 0.0
        return value, value
    if not pred_boundary.any():
        return 0.0, 0.0
    inverse = (1 - pred_boundary) * 255
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    matched = []
    for row in gt_rows:
        values = distance[row][gt_boundary[row].astype(bool)]
        matched.append(bool(values.size and values.min() <= tolerance))
    longest = current = 0
    previous = None
    for row, is_matched in zip(gt_rows.tolist(), matched):
        if is_matched and (previous is None or row == previous + 1):
            current += 1
        elif is_matched:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous = row
    return sum(matched) / len(gt_rows), longest / len(gt_rows)


def mean(items):
    values = [float(item) for item in items if item not in (None, "")]
    return float(np.mean(values)) if values else None


def fmt(value):
    return "" if value is None else f"{value:.6f}"


def summarize(items, model, level, group):
    positives = [item for item in items if item["has_boundary"] == 1]
    negatives = [item for item in items if item["has_boundary"] == 0]
    conditional = [item for item in positives if item["pred_nonempty"] == 1]
    return {
        "model": model,
        "level": level,
        "group": group,
        "samples": len(items),
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "all_iou": fmt(mean(item["iou"] for item in items)),
        "all_dice": fmt(mean(item["dice"] for item in items)),
        "positive_iou": fmt(mean(item["iou"] for item in positives)),
        "positive_dice": fmt(mean(item["dice"] for item in positives)),
        "positive_empty_rate": fmt(
            mean(1 - item["pred_nonempty"] for item in positives)
        ),
        "negative_fpr": fmt(mean(item["pred_nonempty"] for item in negatives)),
        "positive_bf1": fmt(mean(item["bf1"] for item in positives)),
        "pmbe": fmt(mean(item["mbe"] for item in positives)),
        "cmbe": fmt(mean(item["mbe"] for item in conditional)),
        "pmaxbe": fmt(mean(item["maxbe"] for item in positives)),
        "cmaxbe": fmt(mean(item["maxbe"] for item in conditional)),
        "coverage": fmt(mean(item["coverage"] for item in positives)),
        "continuity": fmt(mean(item["continuity"] for item in positives)),
    }


SUMMARY_FIELDS = [
    "model", "level", "group", "samples", "positive_samples",
    "negative_samples", "all_iou", "all_dice", "positive_iou",
    "positive_dice", "positive_empty_rate", "negative_fpr",
    "positive_bf1", "pmbe", "cmbe", "pmaxbe", "cmaxbe", "coverage",
    "continuity",
]


def macro_summarize(scene_summaries, model, group, scene_filter=None):
    selected = [
        row for row in scene_summaries
        if scene_filter is None or scene_filter(row)
    ]
    result = {
        "model": model,
        "level": "scene_macro",
        "group": group,
        "samples": sum(int(row["samples"]) for row in selected),
        "positive_samples": sum(int(row["positive_samples"]) for row in selected),
        "negative_samples": sum(int(row["negative_samples"]) for row in selected),
    }
    for field in SUMMARY_FIELDS[6:]:
        result[field] = fmt(mean(row[field] for row in selected))
    return result


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    split_rows = read_csv(root / "splits" / f"{args.split}.csv")
    models = []
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"Invalid --model value: {item}")
        name, directory = item.split("=", 1)
        models.append((name, Path(directory)))

    per_frame = []
    all_frame_summaries = []
    all_scene_summaries = []
    all_scene_macros = []
    all_subset_summaries = []
    missing = []

    for model, pred_dir in models:
        model_items = []
        for row in split_rows:
            sample_id = row["sample_id"]
            gt = load_mask(root / row["band_mask"], 128)
            pred_path = find_prediction(pred_dir, sample_id)
            if pred_path is None:
                missing.append((model, sample_id))
                pred = np.zeros_like(gt)
            else:
                pred = align_prediction(load_mask(pred_path, args.threshold), gt)
            intersection = int(np.logical_and(pred, gt).sum())
            union = int(np.logical_or(pred, gt).sum())
            total = int(pred.sum() + gt.sum())
            iou = 1.0 if union == 0 else intersection / union
            dice = 1.0 if total == 0 else 2.0 * intersection / total
            bf1, mbe, maxbe = boundary_metrics(pred, gt, args.tolerance)
            coverage, continuity = continuity_metrics(pred, gt, args.tolerance)
            item = {
                "model": model,
                "sample_id": sample_id,
                "scene": row["reviewed_segment_id"],
                "prefix": row["prefix"],
                "subset": row["scene_subset"],
                "has_boundary": int(row["has_boundary"]),
                "pred_nonempty": int(bool(pred.any())),
                "gt_pixels": int(gt.sum()),
                "pred_pixels": int(pred.sum()),
                "iou": iou,
                "dice": dice,
                "bf1": bf1,
                "mbe": mbe,
                "maxbe": maxbe,
                "coverage": coverage,
                "continuity": continuity,
            }
            model_items.append(item)
            per_frame.append({
                key: (f"{value:.6f}" if isinstance(value, float) else value)
                for key, value in item.items()
            })

        all_frame_summaries.append(
            summarize(model_items, model, "frame_macro", "overall")
        )
        scene_summaries = []
        scenes = sorted({item["scene"] for item in model_items})
        for scene in scenes:
            items = [item for item in model_items if item["scene"] == scene]
            row = summarize(items, model, "scene", scene)
            row["subset"] = items[0]["subset"]
            scene_summaries.append(row)
            all_scene_summaries.append(row)
        all_scene_macros.append(
            macro_summarize(scene_summaries, model, "overall")
        )
        for subset in ("positive", "hard_negative", "ood_negative"):
            subset_items = [item for item in model_items if item["subset"] == subset]
            all_subset_summaries.append(
                summarize(subset_items, model, "frame_macro", subset)
            )
            all_scene_macros.append(
                macro_summarize(
                    scene_summaries,
                    model,
                    subset,
                    lambda scene, subset=subset: scene["subset"] == subset,
                )
            )

    frame_fields = [
        "model", "sample_id", "scene", "prefix", "subset",
        "has_boundary", "pred_nonempty", "gt_pixels", "pred_pixels", "iou",
        "dice", "bf1", "mbe", "maxbe", "coverage", "continuity",
    ]
    write_csv(out_dir / "per_frame_metrics.csv", per_frame, frame_fields)
    write_csv(out_dir / "frame_macro_summary.csv", all_frame_summaries, SUMMARY_FIELDS)
    scene_fields = SUMMARY_FIELDS + ["subset"]
    write_csv(out_dir / "per_scene_metrics.csv", all_scene_summaries, scene_fields)
    write_csv(out_dir / "scene_macro_summary.csv", all_scene_macros, SUMMARY_FIELDS)
    write_csv(out_dir / "subset_frame_summary.csv", all_subset_summaries, SUMMARY_FIELDS)
    if missing:
        write_csv(
            out_dir / "missing_predictions.csv",
            ({"model": model, "sample_id": sample} for model, sample in missing),
            ["model", "sample_id"],
        )
        raise RuntimeError(f"Missing {len(missing)} predictions; see output CSV.")
    print(f"Evaluated {len(models)} models on {len(split_rows)} frozen samples.")
    print(f"Saved summaries to: {out_dir}")


if __name__ == "__main__":
    main()
