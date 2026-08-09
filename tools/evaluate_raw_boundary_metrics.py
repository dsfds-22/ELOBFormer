import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_rows(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def image_id_from_row(row):
    return Path(row["image"]).stem


def prefix_from_id(sample_id):
    return sample_id.split("_", 1)[0]


def find_pred_mask(pred_dir, sample_id):
    candidates = [
        pred_dir / f"{sample_id}.png",
        pred_dir / f"{sample_id}_prob.png",
        pred_dir / f"{sample_id}_mask.png",
        pred_dir / f"{sample_id}_pred.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(pred_dir.glob(f"{sample_id}*.png"))
    return matches[0] if matches else None


def load_binary_mask(path, threshold):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask >= threshold).astype(np.uint8)


def align_pred_to_gt(pred_mask, gt_mask):
    """Align prediction to GT size without distorting padded predictions."""

    if pred_mask.shape == gt_mask.shape:
        return pred_mask
    gt_h, gt_w = gt_mask.shape
    pred_h, pred_w = pred_mask.shape
    if pred_h >= gt_h and pred_w >= gt_w:
        return pred_mask[:gt_h, :gt_w]
    return cv2.resize(pred_mask, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)


def mask_boundary(mask, kernel_size=3):
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = (mask.astype(np.uint8) - eroded) > 0
    return boundary.astype(np.uint8)


def nearest_distances(source_boundary, target_boundary):
    if source_boundary.sum() == 0:
        return np.array([], dtype=np.float32)
    if target_boundary.sum() == 0:
        height, width = source_boundary.shape
        diag = math.hypot(height, width)
        return np.full(int(source_boundary.sum()), diag, dtype=np.float32)

    target_inv = (1 - target_boundary.astype(np.uint8)) * 255
    dist = cv2.distanceTransform(target_inv, cv2.DIST_L2, 3)
    return dist[source_boundary.astype(bool)].astype(np.float32)


def boundary_metrics(pred_mask, gt_mask, tolerance):
    pred_boundary = mask_boundary(pred_mask)
    gt_boundary = mask_boundary(gt_mask)
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())

    if pred_count == 0 and gt_count == 0:
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "mean_boundary_error": 0.0,
            "max_boundary_error": 0.0,
            "pred_boundary_pixels": 0,
            "gt_boundary_pixels": 0,
        }
    if pred_count == 0 or gt_count == 0:
        height, width = pred_mask.shape
        diag = math.hypot(height, width)
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "mean_boundary_error": diag,
            "max_boundary_error": diag,
            "pred_boundary_pixels": pred_count,
            "gt_boundary_pixels": gt_count,
        }

    pred_to_gt = nearest_distances(pred_boundary, gt_boundary)
    gt_to_pred = nearest_distances(gt_boundary, pred_boundary)

    precision = float((pred_to_gt <= tolerance).mean()) if pred_to_gt.size else 0.0
    recall = float((gt_to_pred <= tolerance).mean()) if gt_to_pred.size else 0.0
    denom = precision + recall
    f1 = 0.0 if denom == 0 else float(2.0 * precision * recall / denom)

    both = np.concatenate([pred_to_gt, gt_to_pred])
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "mean_boundary_error": float(both.mean()) if both.size else 0.0,
        "max_boundary_error": float(both.max()) if both.size else 0.0,
        "pred_boundary_pixels": pred_count,
        "gt_boundary_pixels": gt_count,
    }


def boundary_continuity_metrics(pred_mask, gt_mask, tolerance):
    """Measure matched-row coverage and the longest continuous matched run.

    A GT-valid row is matched when at least one GT boundary pixel is within
    ``tolerance`` pixels of the predicted boundary. Continuity is the longest
    run of consecutive matched image rows divided by the GT-valid row count.
    """

    pred_boundary = mask_boundary(pred_mask)
    gt_boundary = mask_boundary(gt_mask)
    gt_rows = np.flatnonzero(gt_boundary.any(axis=1))

    if gt_rows.size == 0:
        value = 1.0 if pred_boundary.sum() == 0 else 0.0
        return {
            "row_coverage": value,
            "continuity_rate": value,
            "gt_valid_rows": 0,
            "matched_rows": 0,
            "longest_matched_run": 0,
        }
    if pred_boundary.sum() == 0:
        return {
            "row_coverage": 0.0,
            "continuity_rate": 0.0,
            "gt_valid_rows": int(gt_rows.size),
            "matched_rows": 0,
            "longest_matched_run": 0,
        }

    pred_inv = (1 - pred_boundary.astype(np.uint8)) * 255
    distance_to_prediction = cv2.distanceTransform(
        pred_inv, cv2.DIST_L2, 3
    )
    matched_flags = []
    for row in gt_rows:
        row_distances = distance_to_prediction[row][
            gt_boundary[row].astype(bool)
        ]
        matched_flags.append(
            bool(row_distances.size and row_distances.min() <= tolerance)
        )

    matched_count = int(sum(matched_flags))
    longest_run = 0
    current_run = 0
    previous_row = None
    for row, matched in zip(gt_rows.tolist(), matched_flags):
        if matched and (previous_row is None or row == previous_row + 1):
            current_run += 1
        elif matched:
            current_run = 1
        else:
            current_run = 0
        longest_run = max(longest_run, current_run)
        previous_row = row

    return {
        "row_coverage": matched_count / float(gt_rows.size),
        "continuity_rate": longest_run / float(gt_rows.size),
        "gt_valid_rows": int(gt_rows.size),
        "matched_rows": matched_count,
        "longest_matched_run": int(longest_run),
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, key=None):
    groups = {}
    for row in rows:
        group_key = row[key] if key else "overall"
        groups.setdefault(group_key, []).append(row)

    summary_rows = []
    for group_key, items in sorted(groups.items()):
        positives = [r for r in items if int(r["has_boundary"]) == 1]
        negatives = [r for r in items if int(r["has_boundary"]) == 0]
        neg_fp = [r for r in negatives if int(r["pred_pixels"]) > 0]
        summary_rows.append(
            {
                "group": group_key,
                "samples": len(items),
                "positive_samples": len(positives),
                "negative_samples": len(negatives),
                "boundary_f1_positive": f"{mean(float(r['boundary_f1']) for r in positives):.6f}",
                "boundary_precision_positive": f"{mean(float(r['boundary_precision']) for r in positives):.6f}",
                "boundary_recall_positive": f"{mean(float(r['boundary_recall']) for r in positives):.6f}",
                "mean_boundary_error_positive": f"{mean(float(r['mean_boundary_error']) for r in positives):.6f}",
                "max_boundary_error_positive_mean": f"{mean(float(r['max_boundary_error']) for r in positives):.6f}",
                "row_coverage_positive": f"{mean(float(r['row_coverage']) for r in positives):.6f}",
                "continuity_rate_positive": f"{mean(float(r['continuity_rate']) for r in positives):.6f}",
                "negative_false_positive_rate": f"{(len(neg_fp) / len(negatives)) if negatives else 0.0:.6f}",
            }
        )
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw boundary metrics from predicted masks.")
    parser.add_argument(
        "--dataset-root",
        default="datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601",
    )
    parser.add_argument("--split", default="test", help="Split name, e.g. test/val/train.")
    parser.add_argument("--split-csv", default="", help="Optional explicit split CSV path.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted mask PNGs.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=5.0, help="Boundary matching tolerance in pixels.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    split_csv = Path(args.split_csv) if args.split_csv else dataset_root / "splits" / f"{args.split}.csv"
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    rows = read_rows(split_csv)
    result_rows = []
    missing = []

    for row in rows:
        sample_id = image_id_from_row(row)
        gt_path = dataset_root / row["band_mask"]
        pred_path = find_pred_mask(pred_dir, sample_id)
        if pred_path is None:
            missing.append(sample_id)
            pred_mask = np.zeros((int(row["height"]), int(row["width"])), dtype=np.uint8)
        else:
            pred_mask = load_binary_mask(pred_path, args.threshold)

        gt_mask = load_binary_mask(gt_path, 128)
        pred_mask = align_pred_to_gt(pred_mask, gt_mask)

        metrics = boundary_metrics(pred_mask, gt_mask, args.tolerance)
        continuity = boundary_continuity_metrics(
            pred_mask, gt_mask, args.tolerance
        )
        result = {
            "id": sample_id,
            "prefix": prefix_from_id(sample_id),
            "has_boundary": int(row["has_boundary"]),
            "gt_pixels": int(gt_mask.sum()),
            "pred_pixels": int(pred_mask.sum()),
            **{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in metrics.items()},
            **{
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in continuity.items()
            },
        }
        result_rows.append(result)

    fields = [
        "id",
        "prefix",
        "has_boundary",
        "gt_pixels",
        "pred_pixels",
        "pred_boundary_pixels",
        "gt_boundary_pixels",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "mean_boundary_error",
        "max_boundary_error",
        "gt_valid_rows",
        "matched_rows",
        "longest_matched_run",
        "row_coverage",
        "continuity_rate",
    ]
    write_csv(out_dir / "raw_boundary_metrics.csv", result_rows, fields)

    summary_fields = [
        "group",
        "samples",
        "positive_samples",
        "negative_samples",
        "boundary_f1_positive",
        "boundary_precision_positive",
        "boundary_recall_positive",
        "mean_boundary_error_positive",
        "max_boundary_error_positive_mean",
        "row_coverage_positive",
        "continuity_rate_positive",
        "negative_false_positive_rate",
    ]
    overall = summarize(result_rows)
    by_prefix = summarize(result_rows, "prefix")
    write_csv(out_dir / "raw_boundary_summary.csv", overall, summary_fields)
    write_csv(out_dir / "raw_boundary_prefix_metrics.csv", by_prefix, summary_fields)

    report_lines = [
        "# Raw Boundary Metrics Report",
        "",
        f"- Split CSV: `{split_csv}`",
        f"- Prediction directory: `{pred_dir}`",
        f"- Boundary tolerance: {args.tolerance:.1f} px",
        f"- Threshold: {args.threshold}",
        f"- Samples: {len(result_rows)}",
        f"- Missing predictions: {len(missing)}",
        "",
        "## Overall",
        "",
        "| Samples | Positive | Negative | Boundary F1 | Boundary Precision | Boundary Recall | Mean Boundary Error | Max Boundary Error Mean | Row Coverage | Continuity | Negative FP Rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    item = overall[0]
    report_lines.append(
        f"| {item['samples']} | {item['positive_samples']} | {item['negative_samples']} | "
        f"{item['boundary_f1_positive']} | {item['boundary_precision_positive']} | {item['boundary_recall_positive']} | "
        f"{item['mean_boundary_error_positive']} | {item['max_boundary_error_positive_mean']} | "
        f"{item['row_coverage_positive']} | {item['continuity_rate_positive']} | "
        f"{item['negative_false_positive_rate']} |"
    )
    report_lines.extend(
        [
            "",
            "## Metric Meaning",
            "",
            "- Boundary F1: extracts the mask boundary, then checks whether predicted boundary pixels and GT boundary pixels match within the tolerance radius.",
            "- Mean Boundary Error: symmetric average nearest-pixel distance between predicted boundary and GT boundary. Lower is better.",
            "- Max Boundary Error: symmetric maximum nearest-pixel distance between predicted boundary and GT boundary. Lower is better and reflects worst local deviation.",
            "- Row Coverage: ratio of GT-valid rows that contain a GT boundary pixel within the tolerance of the prediction.",
            "- Continuity: longest consecutive run of matched GT rows divided by the number of GT-valid rows. Higher is better.",
            "- Negative FP Rate: among images without GT boundary, the ratio where the model still predicted non-empty masks.",
        ]
    )
    (out_dir / "raw_boundary_metrics_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    if missing:
        (out_dir / "missing_predictions.txt").write_text("\n".join(missing), encoding="utf-8")


if __name__ == "__main__":
    main()
