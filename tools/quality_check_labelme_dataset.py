import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

EXPECTED_SHAPE_TYPES = {
    "lane1": {"linestrip"},
    "lane2": {"linestrip"},
    "lane3": {"linestrip"},
    "lane4": {"linestrip"},
    "right_boundary": {"linestrip"},
    "emergency_outer_boundary": {"linestrip"},
    "cone": {"rectangle"},
}

LINE_LABELS = {"lane1", "lane2", "lane3", "lane4", "right_boundary", "emergency_outer_boundary"}
CORE_LANE_LABELS = ["lane1", "lane2", "lane3", "lane4"]
BOUNDARY_LABELS = ["right_boundary", "emergency_outer_boundary"]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_size(path):
    with Image.open(path) as img:
        return img.size


def find_image_for_json(json_path, image_path_value):
    candidates = []
    if image_path_value:
        candidates.append(json_path.parent / image_path_value)
        candidates.append(json_path.parent / Path(image_path_value).name)
    for ext in IMAGE_EXTS:
        candidates.append(json_path.with_suffix(ext))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def polyline_length(points):
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    return total


def inspect_dataset(root):
    root = Path(root)
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    jsons = sorted(root.rglob("*.json"))

    image_stems = defaultdict(list)
    for image in images:
        image_stems[image.with_suffix("").name].append(image)

    label_counts = Counter()
    shape_type_counts = Counter()
    label_shape_counts = Counter()
    image_size_counts = Counter()
    json_declared_size_counts = Counter()
    per_image_label_presence = Counter()

    issues = defaultdict(list)
    line_lengths = defaultdict(list)
    total_shapes = 0
    json_with_image_data = 0
    json_without_shapes = 0

    matched_images = set()

    for json_path in jsons:
        rel_json = str(json_path.relative_to(root))
        try:
            data = load_json(json_path)
        except Exception as exc:
            issues["bad_json"].append({"file": rel_json, "detail": str(exc)})
            continue

        if data.get("imageData"):
            json_with_image_data += 1

        image_path = find_image_for_json(json_path, data.get("imagePath"))
        if image_path is None:
            issues["missing_image_for_json"].append({"file": rel_json, "detail": data.get("imagePath", "")})
            continue
        matched_images.add(image_path.resolve())

        try:
            actual_w, actual_h = image_size(image_path)
            image_size_counts[f"{actual_w}x{actual_h}"] += 1
        except Exception as exc:
            issues["bad_image"].append({"file": str(image_path.relative_to(root)), "detail": str(exc)})
            continue

        declared_w = data.get("imageWidth")
        declared_h = data.get("imageHeight")
        json_declared_size_counts[f"{declared_w}x{declared_h}"] += 1
        if declared_w != actual_w or declared_h != actual_h:
            issues["size_mismatch"].append(
                {
                    "file": rel_json,
                    "detail": f"json={declared_w}x{declared_h}, image={actual_w}x{actual_h}",
                }
            )

        shapes = data.get("shapes", [])
        if not shapes:
            json_without_shapes += 1
            issues["empty_shapes"].append({"file": rel_json, "detail": "no shapes"})
            continue

        per_file_counts = Counter()
        for idx, shape in enumerate(shapes):
            total_shapes += 1
            label = str(shape.get("label", "")).strip()
            shape_type = str(shape.get("shape_type", "")).strip()
            points = shape.get("points", [])

            label_counts[label] += 1
            shape_type_counts[shape_type] += 1
            label_shape_counts[(label, shape_type)] += 1
            per_file_counts[label] += 1

            expected = EXPECTED_SHAPE_TYPES.get(label)
            if expected is None:
                issues["unknown_label"].append({"file": rel_json, "detail": f"shape#{idx}: label={label}"})
            elif shape_type not in expected:
                issues["unexpected_shape_type"].append(
                    {
                        "file": rel_json,
                        "detail": f"shape#{idx}: {label} uses {shape_type}, expected {sorted(expected)}",
                    }
                )

            if not isinstance(points, list):
                issues["bad_points"].append({"file": rel_json, "detail": f"shape#{idx}: points is not list"})
                continue

            min_points = 2
            if shape_type == "polygon":
                min_points = 3
            if len(points) < min_points:
                issues["too_few_points"].append(
                    {"file": rel_json, "detail": f"shape#{idx}: {label}/{shape_type} has {len(points)} points"}
                )

            for point_idx, point in enumerate(points):
                if not isinstance(point, list) or len(point) < 2 or not is_number(point[0]) or not is_number(point[1]):
                    issues["bad_points"].append(
                        {"file": rel_json, "detail": f"shape#{idx} point#{point_idx}: invalid point={point}"}
                    )
                    continue
                x = float(point[0])
                y = float(point[1])
                if x < -1 or y < -1 or x > actual_w + 1 or y > actual_h + 1:
                    issues["points_out_of_bounds"].append(
                        {
                            "file": rel_json,
                            "detail": f"shape#{idx} {label} point#{point_idx}=({x:.1f},{y:.1f}) outside {actual_w}x{actual_h}",
                        }
                    )

            if shape_type == "linestrip" and len(points) >= 2:
                length = polyline_length(points)
                line_lengths[label].append(length)
                if label in LINE_LABELS and length < 30:
                    issues["very_short_line"].append(
                        {"file": rel_json, "detail": f"shape#{idx}: {label} length={length:.1f}px"}
                    )
                if label in BOUNDARY_LABELS:
                    mean_x = sum(float(p[0]) for p in points if len(p) >= 2 and is_number(p[0])) / len(points)
                    if mean_x < actual_w * 0.35:
                        issues["boundary_too_left"].append(
                            {"file": rel_json, "detail": f"shape#{idx}: {label} mean_x={mean_x:.1f}px"}
                        )

        for label in per_file_counts:
            per_image_label_presence[label] += 1

        for label in CORE_LANE_LABELS:
            if per_file_counts[label] > 1:
                issues["duplicate_lane_label"].append(
                    {"file": rel_json, "detail": f"{label} appears {per_file_counts[label]} times"}
                )
        for label in BOUNDARY_LABELS:
            if per_file_counts[label] > 1:
                issues["duplicate_boundary_label"].append(
                    {"file": rel_json, "detail": f"{label} appears {per_file_counts[label]} times"}
                )

    unmatched_images = []
    for image in images:
        if image.resolve() not in matched_images:
            possible_json = image.with_suffix(".json")
            if not possible_json.exists():
                unmatched_images.append(str(image.relative_to(root)))

    if unmatched_images:
        for item in unmatched_images:
            issues["missing_json_for_image"].append({"file": item, "detail": "no matching json"})

    missing_label_counts = {}
    for label in CORE_LANE_LABELS + BOUNDARY_LABELS:
        missing_label_counts[label] = len(jsons) - per_image_label_presence.get(label, 0)

    length_summary = {}
    for label, values in line_lengths.items():
        if not values:
            continue
        sorted_values = sorted(values)
        n = len(sorted_values)
        length_summary[label] = {
            "count": n,
            "min": round(sorted_values[0], 2),
            "p50": round(sorted_values[n // 2], 2),
            "mean": round(sum(sorted_values) / n, 2),
            "max": round(sorted_values[-1], 2),
        }

    return {
        "root": str(root),
        "image_count": len(images),
        "json_count": len(jsons),
        "total_shapes": total_shapes,
        "json_with_image_data": json_with_image_data,
        "json_without_shapes": json_without_shapes,
        "image_size_counts": dict(image_size_counts),
        "json_declared_size_counts": dict(json_declared_size_counts),
        "label_counts": dict(label_counts),
        "shape_type_counts": dict(shape_type_counts),
        "label_shape_counts": {f"{k[0]}|{k[1]}": v for k, v in label_shape_counts.items()},
        "per_image_label_presence": dict(per_image_label_presence),
        "missing_label_counts": missing_label_counts,
        "line_length_summary": length_summary,
        "issues": dict(issues),
    }


def format_issue_table(items, limit=20):
    if not items:
        return "- None\n"
    lines = []
    for item in items[:limit]:
        lines.append(f"- `{item['file']}`: {item['detail']}")
    if len(items) > limit:
        lines.append(f"- ... {len(items) - limit} more")
    return "\n".join(lines) + "\n"


def write_markdown(report, path):
    issue_counts = {name: len(items) for name, items in report["issues"].items()}
    ordered_issue_counts = sorted(issue_counts.items(), key=lambda x: (-x[1], x[0]))

    lines = []
    lines.append("# LabelMe Dataset Quality Report")
    lines.append("")
    lines.append(f"- Root: `{report['root']}`")
    lines.append(f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Images: `{report['image_count']}`")
    lines.append(f"- JSON files: `{report['json_count']}`")
    lines.append(f"- Total shapes: `{report['total_shapes']}`")
    lines.append(f"- JSON with embedded imageData: `{report['json_with_image_data']}`")
    lines.append("")

    lines.append("## Sizes")
    lines.append("")
    lines.append("Actual image sizes:")
    for key, value in sorted(report["image_size_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("JSON declared sizes:")
    for key, value in sorted(report["json_declared_size_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")

    lines.append("## Labels")
    lines.append("")
    for key, value in sorted(report["label_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")

    lines.append("## Shape Types")
    lines.append("")
    for key, value in sorted(report["shape_type_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")

    lines.append("## Label Presence By Image")
    lines.append("")
    for key, value in sorted(report["per_image_label_presence"].items()):
        missing = report["json_count"] - value
        lines.append(f"- `{key}` present in {value} images, missing in {missing} images")
    lines.append("")

    lines.append("## Missing Core Labels")
    lines.append("")
    for key, value in report["missing_label_counts"].items():
        lines.append(f"- `{key}` missing images: {value}")
    lines.append("")

    lines.append("## Line Length Summary")
    lines.append("")
    for key, stats in sorted(report["line_length_summary"].items()):
        lines.append(
            f"- `{key}`: count={stats['count']}, min={stats['min']}, p50={stats['p50']}, "
            f"mean={stats['mean']}, max={stats['max']}"
        )
    lines.append("")

    lines.append("## Issue Counts")
    lines.append("")
    if ordered_issue_counts:
        for key, value in ordered_issue_counts:
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Issue Samples")
    lines.append("")
    for key, _ in ordered_issue_counts:
        lines.append(f"### {key}")
        lines.append("")
        lines.append(format_issue_table(report["issues"][key]))

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Quality check a LabelMe dataset.")
    parser.add_argument("--root", required=True, help="Dataset root containing images and LabelMe JSON files.")
    parser.add_argument("--out-dir", default="reports/dataset_quality", help="Output directory for reports.")
    parser.add_argument("--name", default=None, help="Report basename. Defaults to timestamp.")
    args = parser.parse_args()

    report = inspect_dataset(args.root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or datetime.now().strftime("quality_report_%Y%m%d_%H%M%S")
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    issue_count = sum(len(items) for items in report["issues"].values())
    print(f"Images: {report['image_count']}")
    print(f"JSON files: {report['json_count']}")
    print(f"Total shapes: {report['total_shapes']}")
    print(f"Issue items: {issue_count}")
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
