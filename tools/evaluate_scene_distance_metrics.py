import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from evaluate_raw_boundary_metrics import (
    align_pred_to_gt,
    boundary_continuity_metrics,
    boundary_metrics,
    find_pred_mask,
    load_binary_mask,
    mask_boundary,
    nearest_distances,
    read_rows,
)


DISTANCE_ORDER = {"far": 0, "middle": 1, "near": 2}


def mean(values):
    values = [float(value) for value in values if value not in ("", None)]
    return sum(values) / len(values) if values else None


def format_value(value):
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: format_value(row.get(field, "")) for field in fieldnames}
            )


def parse_model_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Model must use NAME=PREDICTION_DIRECTORY."
        )
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "Model must use NAME=PREDICTION_DIRECTORY."
        )
    return name.strip(), Path(path.strip())


def load_taxonomy(path):
    if not path:
        return {}
    rows = read_rows(Path(path))
    return {row["prefix"]: row for row in rows}


def sample_id_from_row(row):
    return row.get("sample_id") or Path(row["image"]).stem


def scene_group_from_row(row):
    for key in (
        "reviewed_segment_id",
        "scene_segment",
        "capture_session",
        "prefix",
    ):
        value = row.get(key, "").strip()
        if value:
            return value
    return "unknown"


def overlap_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    total = int(pred.sum() + gt.sum())
    iou = 1.0 if union == 0 else intersection / float(union)
    dice = 1.0 if total == 0 else 2.0 * intersection / float(total)
    return iou, dice


def restricted_boundary(boundary, y_start, y_end):
    result = np.zeros_like(boundary, dtype=np.uint8)
    result[y_start:y_end] = boundary[y_start:y_end]
    return result


def metrics_from_boundaries(pred_boundary, gt_boundary, tolerance):
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    if gt_count == 0:
        return None

    if pred_count == 0:
        height, width = gt_boundary.shape
        diagonal = math.hypot(height, width)
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "mean_boundary_error": diagonal,
            "max_boundary_error": diagonal,
            "pred_boundary_pixels": 0,
            "gt_boundary_pixels": gt_count,
        }

    pred_to_gt = nearest_distances(pred_boundary, gt_boundary)
    gt_to_pred = nearest_distances(gt_boundary, pred_boundary)
    precision = float((pred_to_gt <= tolerance).mean())
    recall = float((gt_to_pred <= tolerance).mean())
    denominator = precision + recall
    combined = np.concatenate([pred_to_gt, gt_to_pred])
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": (
            0.0 if denominator == 0 else 2.0 * precision * recall / denominator
        ),
        "mean_boundary_error": float(combined.mean()),
        "max_boundary_error": float(combined.max()),
        "pred_boundary_pixels": pred_count,
        "gt_boundary_pixels": gt_count,
    }


def segment_continuity(pred_boundary, gt_boundary, tolerance):
    gt_rows = np.flatnonzero(gt_boundary.any(axis=1))
    if gt_rows.size == 0:
        return None
    if pred_boundary.sum() == 0:
        return {
            "row_coverage": 0.0,
            "continuity_rate": 0.0,
            "gt_valid_rows": int(gt_rows.size),
            "matched_rows": 0,
            "longest_matched_run": 0,
        }

    prediction_inverse = (1 - pred_boundary.astype(np.uint8)) * 255
    distances = cv2.distanceTransform(prediction_inverse, cv2.DIST_L2, 3)
    matched = []
    for row in gt_rows:
        row_distances = distances[row][gt_boundary[row].astype(bool)]
        matched.append(
            bool(row_distances.size and row_distances.min() <= tolerance)
        )

    longest = 0
    current = 0
    previous_row = None
    for row, is_matched in zip(gt_rows.tolist(), matched):
        if is_matched and (previous_row is None or row == previous_row + 1):
            current += 1
        elif is_matched:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous_row = row

    matched_count = int(sum(matched))
    return {
        "row_coverage": matched_count / float(gt_rows.size),
        "continuity_rate": longest / float(gt_rows.size),
        "gt_valid_rows": int(gt_rows.size),
        "matched_rows": matched_count,
        "longest_matched_run": int(longest),
    }


def distance_ranges(gt_boundary):
    rows = np.flatnonzero(gt_boundary.any(axis=1))
    if rows.size < 3:
        return []
    edges = np.linspace(int(rows[0]), int(rows[-1]) + 1, 4)
    edges = np.rint(edges).astype(int)
    edges[0] = int(rows[0])
    edges[-1] = int(rows[-1]) + 1
    names = ("far", "middle", "near")
    ranges = []
    for index, name in enumerate(names):
        start = int(edges[index])
        end = int(edges[index + 1])
        if end > start and gt_boundary[start:end].any():
            ranges.append((name, start, end))
    return ranges


def row_center(mask, row):
    columns = np.flatnonzero(mask[row])
    return float(columns.mean()) if columns.size else None


def endpoint_error(pred_mask, gt_mask, use_top, max_row_gap=10):
    gt_rows = np.flatnonzero(gt_mask.any(axis=1))
    pred_rows = np.flatnonzero(pred_mask.any(axis=1))
    if gt_rows.size == 0:
        return None
    if pred_rows.size == 0:
        return math.hypot(*gt_mask.shape)

    gt_row = int(gt_rows[0] if use_top else gt_rows[-1])
    candidate_index = int(np.argmin(np.abs(pred_rows - gt_row)))
    pred_row = int(pred_rows[candidate_index])
    if abs(pred_row - gt_row) > max_row_gap:
        return math.hypot(*gt_mask.shape)

    gt_x = row_center(gt_mask, gt_row)
    pred_x = row_center(pred_mask, pred_row)
    return math.hypot(pred_x - gt_x, pred_row - gt_row)


def evaluate_model(
    model_name,
    pred_dir,
    dataset_root,
    source_rows,
    taxonomy,
    threshold,
    tolerance,
):
    image_rows = []
    distance_rows = []
    missing_predictions = []

    for source in source_rows:
        sample_id = sample_id_from_row(source)
        prefix = source.get("prefix") or sample_id.split("_", 1)[0]
        taxonomy_row = taxonomy.get(prefix, {})
        gt_path = dataset_root / source["band_mask"]
        pred_path = find_pred_mask(pred_dir, sample_id)
        gt_mask = load_binary_mask(gt_path, 128)

        if pred_path is None:
            pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
            missing_predictions.append(sample_id)
        else:
            pred_mask = load_binary_mask(pred_path, threshold)
            pred_mask = align_pred_to_gt(pred_mask, gt_mask)

        has_boundary = int(source["has_boundary"])
        image_diagonal = math.hypot(*gt_mask.shape)
        iou, dice = overlap_metrics(pred_mask, gt_mask)
        full_boundary = boundary_metrics(pred_mask, gt_mask, tolerance)
        full_continuity = boundary_continuity_metrics(
            pred_mask, gt_mask, tolerance
        )
        pred_boundary = mask_boundary(pred_mask)
        gt_boundary = mask_boundary(gt_mask)

        image_rows.append(
            {
                "model": model_name,
                "sample_id": sample_id,
                "prefix": prefix,
                "capture_session": source.get("capture_session", ""),
                "scene_group": scene_group_from_row(source),
                "semantic_scene": taxonomy_row.get(
                    "semantic_scene", "unassigned"
                ),
                "taxonomy_status": taxonomy_row.get("status", "unassigned"),
                "has_boundary": has_boundary,
                "prediction_missing": int(pred_path is None),
                "prediction_empty": int(pred_mask.sum() == 0),
                "gt_pixels": int(gt_mask.sum()),
                "pred_pixels": int(pred_mask.sum()),
                "iou": iou,
                "dice": dice,
                "boundary_precision": full_boundary["boundary_precision"],
                "boundary_recall": full_boundary["boundary_recall"],
                "boundary_f1": full_boundary["boundary_f1"],
                "mean_boundary_error": full_boundary[
                    "mean_boundary_error"
                ],
                "max_boundary_error": full_boundary["max_boundary_error"],
                "penalized_mean_boundary_error": full_boundary[
                    "mean_boundary_error"
                ],
                "conditional_mean_boundary_error": (
                    full_boundary["mean_boundary_error"]
                    if pred_mask.sum() > 0 and gt_mask.sum() > 0
                    else None
                ),
                "penalized_max_boundary_error": full_boundary[
                    "max_boundary_error"
                ],
                "conditional_max_boundary_error": (
                    full_boundary["max_boundary_error"]
                    if pred_mask.sum() > 0 and gt_mask.sum() > 0
                    else None
                ),
                "penalized_normalized_mbe": (
                    full_boundary["mean_boundary_error"] / image_diagonal
                ),
                "conditional_normalized_mbe": (
                    full_boundary["mean_boundary_error"] / image_diagonal
                    if pred_mask.sum() > 0 and gt_mask.sum() > 0
                    else None
                ),
                "row_coverage": full_continuity["row_coverage"],
                "continuity_rate": full_continuity["continuity_rate"],
                "top_endpoint_error": (
                    endpoint_error(pred_mask, gt_mask, True)
                    if has_boundary
                    else None
                ),
                "bottom_endpoint_error": (
                    endpoint_error(pred_mask, gt_mask, False)
                    if has_boundary
                    else None
                ),
            }
        )

        if not has_boundary:
            continue
        for distance_name, y_start, y_end in distance_ranges(gt_boundary):
            pred_part = restricted_boundary(
                pred_boundary, y_start, y_end
            )
            gt_part = restricted_boundary(gt_boundary, y_start, y_end)
            segment_metrics = metrics_from_boundaries(
                pred_part, gt_part, tolerance
            )
            segment_cont = segment_continuity(
                pred_part, gt_part, tolerance
            )
            if segment_metrics is None or segment_cont is None:
                continue
            distance_rows.append(
                {
                    "model": model_name,
                    "sample_id": sample_id,
                    "prefix": prefix,
                    "scene_group": scene_group_from_row(source),
                    "semantic_scene": taxonomy_row.get(
                        "semantic_scene", "unassigned"
                    ),
                    "distance_region": distance_name,
                    "y_start": y_start,
                    "y_end": y_end,
                    "prediction_empty": int(pred_part.sum() == 0),
                    "penalized_mean_boundary_error": segment_metrics[
                        "mean_boundary_error"
                    ],
                    "conditional_mean_boundary_error": (
                        segment_metrics["mean_boundary_error"]
                        if pred_part.sum() > 0
                        else None
                    ),
                    "penalized_max_boundary_error": segment_metrics[
                        "max_boundary_error"
                    ],
                    "conditional_max_boundary_error": (
                        segment_metrics["max_boundary_error"]
                        if pred_part.sum() > 0
                        else None
                    ),
                    "penalized_normalized_mbe": (
                        segment_metrics["mean_boundary_error"]
                        / image_diagonal
                    ),
                    "conditional_normalized_mbe": (
                        segment_metrics["mean_boundary_error"]
                        / image_diagonal
                        if pred_part.sum() > 0
                        else None
                    ),
                    **segment_metrics,
                    **segment_cont,
                }
            )

    return image_rows, distance_rows, missing_predictions


def summarize_images(rows, group_field):
    groups = {}
    for row in rows:
        key = (row["model"], row[group_field])
        groups.setdefault(key, []).append(row)

    output = []
    for (model, group), items in sorted(groups.items()):
        positives = [row for row in items if row["has_boundary"] == 1]
        negatives = [row for row in items if row["has_boundary"] == 0]
        detected_positives = [
            row for row in positives if row["prediction_empty"] == 0
        ]
        output.append(
            {
                "model": model,
                "group": group,
                "samples": len(items),
                "positive_samples": len(positives),
                "negative_samples": len(negatives),
                "all_iou": mean(row["iou"] for row in items),
                "all_dice": mean(row["dice"] for row in items),
                "positive_iou": mean(row["iou"] for row in positives),
                "positive_dice": mean(row["dice"] for row in positives),
                "positive_zero_prediction_rate": (
                    mean(row["prediction_empty"] for row in positives)
                ),
                "negative_false_positive_rate": (
                    mean(
                        int(row["pred_pixels"] > 0)
                        for row in negatives
                    )
                ),
                "boundary_f1_positive": mean(
                    row["boundary_f1"] for row in positives
                ),
                "boundary_precision_positive": mean(
                    row["boundary_precision"] for row in positives
                ),
                "boundary_recall_positive": mean(
                    row["boundary_recall"] for row in positives
                ),
                "mean_boundary_error_positive": mean(
                    row["mean_boundary_error"] for row in positives
                ),
                "max_boundary_error_positive": mean(
                    row["max_boundary_error"] for row in positives
                ),
                "pmbe_positive": mean(
                    row["penalized_mean_boundary_error"]
                    for row in positives
                ),
                "cmbe_positive": mean(
                    row["conditional_mean_boundary_error"]
                    for row in detected_positives
                ),
                "pmaxbe_positive": mean(
                    row["penalized_max_boundary_error"]
                    for row in positives
                ),
                "cmaxbe_positive": mean(
                    row["conditional_max_boundary_error"]
                    for row in detected_positives
                ),
                "pnmbe_positive": mean(
                    row["penalized_normalized_mbe"]
                    for row in positives
                ),
                "cnmbe_positive": mean(
                    row["conditional_normalized_mbe"]
                    for row in detected_positives
                ),
                "row_coverage_positive": mean(
                    row["row_coverage"] for row in positives
                ),
                "continuity_positive": mean(
                    row["continuity_rate"] for row in positives
                ),
                "top_endpoint_error_positive": mean(
                    row["top_endpoint_error"] for row in positives
                ),
                "bottom_endpoint_error_positive": mean(
                    row["bottom_endpoint_error"] for row in positives
                ),
            }
        )
    return output


def summarize_distance(rows, group_field=None):
    groups = {}
    for row in rows:
        group = row[group_field] if group_field else "overall"
        key = (row["model"], group, row["distance_region"])
        groups.setdefault(key, []).append(row)

    output = []
    for (model, group, region), items in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            DISTANCE_ORDER[item[0][2]],
        ),
    ):
        detected_items = [
            row for row in items if row["prediction_empty"] == 0
        ]
        output.append(
            {
                "model": model,
                "group": group,
                "distance_region": region,
                "samples": len(items),
                "prediction_empty_rate": mean(
                    row["prediction_empty"] for row in items
                ),
                "boundary_f1": mean(
                    row["boundary_f1"] for row in items
                ),
                "boundary_precision": mean(
                    row["boundary_precision"] for row in items
                ),
                "boundary_recall": mean(
                    row["boundary_recall"] for row in items
                ),
                "mean_boundary_error": mean(
                    row["mean_boundary_error"] for row in items
                ),
                "max_boundary_error": mean(
                    row["max_boundary_error"] for row in items
                ),
                "pmbe": mean(
                    row["penalized_mean_boundary_error"] for row in items
                ),
                "cmbe": mean(
                    row["conditional_mean_boundary_error"]
                    for row in detected_items
                ),
                "pmaxbe": mean(
                    row["penalized_max_boundary_error"] for row in items
                ),
                "cmaxbe": mean(
                    row["conditional_max_boundary_error"]
                    for row in detected_items
                ),
                "pnmbe": mean(
                    row["penalized_normalized_mbe"] for row in items
                ),
                "cnmbe": mean(
                    row["conditional_normalized_mbe"]
                    for row in detected_items
                ),
                "row_coverage": mean(
                    row["row_coverage"] for row in items
                ),
                "continuity": mean(
                    row["continuity_rate"] for row in items
                ),
            }
        )
    return output


def prefix_macro(prefix_rows):
    by_model = {}
    for row in prefix_rows:
        by_model.setdefault(row["model"], []).append(row)

    output = []
    metrics = (
        "all_iou",
        "all_dice",
        "positive_iou",
        "positive_dice",
        "positive_zero_prediction_rate",
        "negative_false_positive_rate",
        "boundary_f1_positive",
        "mean_boundary_error_positive",
        "max_boundary_error_positive",
        "pmbe_positive",
        "cmbe_positive",
        "pmaxbe_positive",
        "cmaxbe_positive",
        "pnmbe_positive",
        "cnmbe_positive",
        "row_coverage_positive",
        "continuity_positive",
    )
    for model, items in sorted(by_model.items()):
        result = {"model": model, "prefixes": len(items)}
        for metric in metrics:
            result[metric] = mean(row[metric] for row in items)
        output.append(result)
    return output


def build_worst_cases(image_rows, limit):
    output = []
    by_model = {}
    for row in image_rows:
        by_model.setdefault(row["model"], []).append(row)

    for model, rows in sorted(by_model.items()):
        positives = [row for row in rows if row["has_boundary"] == 1]
        negatives = [row for row in rows if row["has_boundary"] == 0]
        criteria = [
            (
                "highest_mean_boundary_error",
                sorted(
                    positives,
                    key=lambda row: row["mean_boundary_error"],
                    reverse=True,
                )[:limit],
            ),
            (
                "lowest_boundary_f1",
                sorted(positives, key=lambda row: row["boundary_f1"])[
                    :limit
                ],
            ),
            (
                "false_positive_negative",
                [
                    row
                    for row in sorted(
                        negatives,
                        key=lambda row: row["pred_pixels"],
                        reverse=True,
                    )
                    if row["pred_pixels"] > 0
                ][:limit],
            ),
        ]
        for criterion, selected in criteria:
            for rank, row in enumerate(selected, start=1):
                output.append(
                    {
                        "model": model,
                        "criterion": criterion,
                        "rank": rank,
                        "sample_id": row["sample_id"],
                        "prefix": row["prefix"],
                        "semantic_scene": row["semantic_scene"],
                        "iou": row["iou"],
                        "boundary_f1": row["boundary_f1"],
                        "mean_boundary_error": row[
                            "mean_boundary_error"
                        ],
                        "max_boundary_error": row[
                            "max_boundary_error"
                        ],
                        "pred_pixels": row["pred_pixels"],
                    }
                )
    return output


IMAGE_FIELDS = [
    "model",
    "sample_id",
    "prefix",
    "capture_session",
    "scene_group",
    "semantic_scene",
    "taxonomy_status",
    "has_boundary",
    "prediction_missing",
    "prediction_empty",
    "gt_pixels",
    "pred_pixels",
    "iou",
    "dice",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "mean_boundary_error",
    "max_boundary_error",
    "penalized_mean_boundary_error",
    "conditional_mean_boundary_error",
    "penalized_max_boundary_error",
    "conditional_max_boundary_error",
    "penalized_normalized_mbe",
    "conditional_normalized_mbe",
    "row_coverage",
    "continuity_rate",
    "top_endpoint_error",
    "bottom_endpoint_error",
]

SUMMARY_FIELDS = [
    "model",
    "group",
    "samples",
    "positive_samples",
    "negative_samples",
    "all_iou",
    "all_dice",
    "positive_iou",
    "positive_dice",
    "positive_zero_prediction_rate",
    "negative_false_positive_rate",
    "boundary_f1_positive",
    "boundary_precision_positive",
    "boundary_recall_positive",
    "mean_boundary_error_positive",
    "max_boundary_error_positive",
    "pmbe_positive",
    "cmbe_positive",
    "pmaxbe_positive",
    "cmaxbe_positive",
    "pnmbe_positive",
    "cnmbe_positive",
    "row_coverage_positive",
    "continuity_positive",
    "top_endpoint_error_positive",
    "bottom_endpoint_error_positive",
]

DISTANCE_FIELDS = [
    "model",
    "sample_id",
    "prefix",
    "scene_group",
    "semantic_scene",
    "distance_region",
    "y_start",
    "y_end",
    "prediction_empty",
    "pred_boundary_pixels",
    "gt_boundary_pixels",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "mean_boundary_error",
    "max_boundary_error",
    "penalized_mean_boundary_error",
    "conditional_mean_boundary_error",
    "penalized_max_boundary_error",
    "conditional_max_boundary_error",
    "penalized_normalized_mbe",
    "conditional_normalized_mbe",
    "gt_valid_rows",
    "matched_rows",
    "longest_matched_run",
    "row_coverage",
    "continuity_rate",
]

DISTANCE_SUMMARY_FIELDS = [
    "model",
    "group",
    "distance_region",
    "samples",
    "prediction_empty_rate",
    "boundary_f1",
    "boundary_precision",
    "boundary_recall",
    "mean_boundary_error",
    "max_boundary_error",
    "pmbe",
    "cmbe",
    "pmaxbe",
    "cmaxbe",
    "pnmbe",
    "cnmbe",
    "row_coverage",
    "continuity",
]


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_report(
    path,
    dataset_root,
    split_csv,
    model_specs,
    overall,
    prefix_macro_rows,
    distance_summary,
    missing_by_model,
):
    lines = [
        "# Scene and Distance Evaluation",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{dataset_root}`",
        f"- Split: `{split_csv}`",
        "- Input/output size: 800 x 450",
        "- Formal prediction masks are binary 0/255 images exported by `probability >= 0.5`; evaluation loads them with `mask >= 128`",
        "- SegFormer logits are bilinearly resized to 800 x 450 before sigmoid and binarization; no mask resize is normally needed",
        "- If a prediction size does not match GT, the already-binary mask is aligned with nearest-neighbor interpolation",
        "- All IoU/All Dice are computed per image over all positive and negative samples and then macro-averaged; GT/prediction both empty scores 1, while GT empty/prediction non-empty scores 0",
        "- Positive IoU/Dice, BF1, PMBE/CMBE, PMaxBE/CMaxBE, Coverage, and Continuity are reported only on GT-nonempty positive samples",
        "- Boundary matching tolerance: 5 px",
        "- No V2, lane constraint, fitting, or geometry correction",
        "- Far/middle/near are thirds of each positive GT boundary's valid vertical span",
        "- Semantic taxonomy is provisional; prefix and scene-group results are objective",
        "",
        "## Overall",
        "",
    ]
    overall_rows = []
    for row in overall:
        overall_rows.append(
            [
                row["model"],
                f"{row['all_iou']:.4f}",
                f"{row['positive_iou']:.4f}",
                (
                    "n/a"
                    if row["negative_false_positive_rate"] is None
                    else f"{100 * row['negative_false_positive_rate']:.2f}%"
                ),
                f"{row['boundary_f1_positive']:.4f}",
                f"{100 * row['positive_zero_prediction_rate']:.2f}%",
                f"{row['pmbe_positive']:.2f}",
                f"{row['cmbe_positive']:.2f}",
                f"{row['pmaxbe_positive']:.2f}",
                f"{row['cmaxbe_positive']:.2f}",
                f"{row['row_coverage_positive']:.4f}",
                f"{row['continuity_positive']:.4f}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Model",
                "All IoU",
                "Positive IoU",
                "Negative FPR",
                "Boundary F1",
                "Empty Rate",
                "PMBE",
                "CMBE",
                "PMaxBE",
                "CMaxBE",
                "Coverage",
                "Continuity",
            ],
            overall_rows,
        )
    )
    lines.extend(["", "## Prefix Macro Average", ""])
    macro_table = []
    for row in prefix_macro_rows:
        macro_table.append(
            [
                row["model"],
                f"{row['positive_iou']:.4f}",
                f"{row['boundary_f1_positive']:.4f}",
                f"{row['pmbe_positive']:.2f}",
                f"{row['cmbe_positive']:.2f}",
                f"{row['row_coverage_positive']:.4f}",
                f"{row['continuity_positive']:.4f}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Model",
                "Positive IoU",
                "Boundary F1",
                "PMBE",
                "CMBE",
                "Coverage",
                "Continuity",
            ],
            macro_table,
        )
    )
    lines.extend(["", "## Far / Middle / Near", ""])
    distance_table = []
    for row in distance_summary:
        distance_table.append(
            [
                row["model"],
                row["distance_region"],
                str(row["samples"]),
                f"{row['boundary_f1']:.4f}",
                f"{row['pmbe']:.2f}",
                f"{row['cmbe']:.2f}",
                f"{row['row_coverage']:.4f}",
                f"{row['continuity']:.4f}",
                f"{100 * row['prediction_empty_rate']:.2f}%",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Model",
                "Region",
                "Samples",
                "Boundary F1",
                "PMBE",
                "CMBE",
                "Coverage",
                "Continuity",
                "Empty",
            ],
            distance_table,
        )
    )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Positive PMBE is macro-averaged over positive images. It uses the per-image symmetric mean boundary distance and assigns the 800 x 450 image diagonal, 917.877988 px, when prediction is empty.",
            "- Positive CMBE is the same symmetric mean distance conditioned on both GT and prediction being non-empty.",
            "- PMaxBE first computes each positive image's maximum bidirectional boundary distance, assigns 917.877988 px to an empty prediction, and then macro-averages over positive images.",
            "- CMaxBE is the per-image maximum bidirectional distance macro-average conditioned on both GT and prediction being non-empty.",
            "- Negative prediction errors are excluded from the four distance metrics and are evaluated separately by frame-level Negative FPR.",
            "- Coverage is the fraction of GT-valid rows containing a predicted boundary within 5 px; Continuity is the longest consecutive run of such matched rows divided by the number of GT-valid rows.",
            "- Far/middle/near are image-space approximations, not metric-distance bins.",
            "- Coverage prevents a short but accurate prediction from receiving an overly favorable distance score.",
            "- Endpoint errors are stored in `per_image_metrics.csv`.",
            "- Crossed prefix-by-distance results are stored in `near_mid_far_prefix_metrics.csv`.",
            "- `worst_cases.csv` lists high-error positives, low-BF1 positives, and negative false positives.",
            "",
            "## Missing Predictions",
            "",
        ]
    )
    for model, _ in model_specs:
        lines.append(f"- {model}: {len(missing_by_model.get(model, []))}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate raw boundary predictions by scene and distance."
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-csv", default="")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=parse_model_spec,
        help="Repeat NAME=PREDICTION_DIRECTORY for each model.",
    )
    parser.add_argument(
        "--taxonomy",
        default="reports/scene_distance_evaluation/scene_taxonomy.csv",
    )
    parser.add_argument(
        "--out-dir", default="reports/scene_distance_evaluation"
    )
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--worst-limit", type=int, default=20)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    split_csv = (
        Path(args.split_csv)
        if args.split_csv
        else dataset_root / "splits" / f"{args.split}.csv"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(args.taxonomy)
    source_rows = read_rows(split_csv)

    all_images = []
    all_distances = []
    missing_by_model = {}
    for model_name, pred_dir in args.model:
        if not pred_dir.is_dir():
            raise FileNotFoundError(pred_dir)
        image_rows, distance_rows, missing = evaluate_model(
            model_name,
            pred_dir,
            dataset_root,
            source_rows,
            taxonomy,
            args.threshold,
            args.tolerance,
        )
        all_images.extend(image_rows)
        all_distances.extend(distance_rows)
        missing_by_model[model_name] = missing

    prefix_rows = summarize_images(all_images, "prefix")
    scene_group_rows = summarize_images(all_images, "scene_group")
    semantic_rows = summarize_images(all_images, "semantic_scene")
    overall_rows = summarize_images(
        [{**row, "_overall": "overall"} for row in all_images],
        "_overall",
    )
    distance_summary = summarize_distance(all_distances)
    distance_prefix = summarize_distance(all_distances, "prefix")
    distance_semantic = summarize_distance(
        all_distances, "semantic_scene"
    )
    macro_rows = prefix_macro(prefix_rows)
    worst_rows = build_worst_cases(all_images, args.worst_limit)

    write_csv(out_dir / "per_image_metrics.csv", all_images, IMAGE_FIELDS)
    write_csv(
        out_dir / "per_image_distance_metrics.csv",
        all_distances,
        DISTANCE_FIELDS,
    )
    write_csv(out_dir / "overall_metrics.csv", overall_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "prefix_metrics.csv", prefix_rows, SUMMARY_FIELDS)
    write_csv(
        out_dir / "scene_group_metrics.csv",
        scene_group_rows,
        SUMMARY_FIELDS,
    )
    write_csv(
        out_dir / "semantic_scene_metrics.csv",
        semantic_rows,
        SUMMARY_FIELDS,
    )
    write_csv(
        out_dir / "near_mid_far_metrics.csv",
        distance_summary,
        DISTANCE_SUMMARY_FIELDS,
    )
    write_csv(
        out_dir / "near_mid_far_prefix_metrics.csv",
        distance_prefix,
        DISTANCE_SUMMARY_FIELDS,
    )
    write_csv(
        out_dir / "near_mid_far_semantic_metrics.csv",
        distance_semantic,
        DISTANCE_SUMMARY_FIELDS,
    )
    macro_fields = ["model", "prefixes"] + [
        key
        for key in macro_rows[0].keys()
        if key not in ("model", "prefixes")
    ]
    write_csv(out_dir / "prefix_macro_metrics.csv", macro_rows, macro_fields)
    write_csv(
        out_dir / "worst_cases.csv",
        worst_rows,
        [
            "model",
            "criterion",
            "rank",
            "sample_id",
            "prefix",
            "semantic_scene",
            "iou",
            "boundary_f1",
            "mean_boundary_error",
            "max_boundary_error",
            "pred_pixels",
        ],
    )
    write_report(
        out_dir / "evaluation_report.md",
        dataset_root,
        split_csv,
        args.model,
        overall_rows,
        macro_rows,
        distance_summary,
        missing_by_model,
    )

    for model_name, _ in args.model:
        print(
            f"{model_name}: missing predictions="
            f"{len(missing_by_model[model_name])}"
        )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
