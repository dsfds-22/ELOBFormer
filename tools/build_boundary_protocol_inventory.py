import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOUNDARY_LABELS = {"right_boundary", "emergency_outer_boundary"}
CONTINUOUS_PREFIXES = {"021", "022", "023", "024", "025", "026"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a non-destructive inventory and metadata templates for the boundary dataset."
    )
    parser.add_argument("--source-root", default="development_labelme")
    parser.add_argument(
        "--output-root",
        default="datasets/emergency_outer_boundary_protocol_v1",
    )
    parser.add_argument(
        "--contact-step",
        type=int,
        default=0,
        help="Fixed sampling step. Use 0 to choose a per-prefix step automatically.",
    )
    parser.add_argument("--contact-max-samples", type=int, default=60)
    parser.add_argument("--contact-columns", type=int, default=5)
    parser.add_argument("--contact-rows", type=int, default=4)
    parser.add_argument("--thumbnail-width", type=int, default=240)
    parser.add_argument("--thumbnail-height", type=int, default=135)
    return parser.parse_args()


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def prefix_of(stem):
    return stem.split("_", 1)[0]


def frame_number_of(stem):
    match = re.match(r"^[^_]+_(\d+)", stem)
    return int(match.group(1)) if match else -1


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def inspect_sample(image_path, json_path, source_split):
    stem = image_path.stem
    prefix = prefix_of(stem)
    capture_session = (
        "session_021_026"
        if source_split == "train_val" and prefix in CONTINUOUS_PREFIXES
        else f"session_{prefix}"
    )
    scene_segment = (
        "session_021_026_pending"
        if capture_session == "session_021_026"
        else f"{capture_session}_scene_001"
    )

    try:
        with Image.open(image_path) as image:
            width, height = image.size
        image_error = ""
    except Exception as exc:
        width, height = 0, 0
        image_error = f"{type(exc).__name__}: {exc}"

    data, json_error = read_json(json_path) if json_path.exists() else (None, "missing_json")
    labels = []
    shape_types = []
    boundary_count = 0
    json_width = ""
    json_height = ""
    if data is not None:
        json_width = data.get("imageWidth", "")
        json_height = data.get("imageHeight", "")
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            labels.append(label)
            shape_types.append(str(shape.get("shape_type", "")).strip())
            if label in BOUNDARY_LABELS:
                boundary_count += 1

    return {
        "sample_id": stem,
        "source_split": source_split,
        "prefix": prefix,
        "capture_session": capture_session,
        "scene_segment": scene_segment,
        "source_domain": capture_session,
        "frame_order": frame_number_of(stem),
        "image_path": str(image_path.resolve()),
        "json_path": str(json_path.resolve()),
        "width": width,
        "height": height,
        "json_width": json_width,
        "json_height": json_height,
        "has_boundary": int(boundary_count > 0),
        "boundary_polyline_count": boundary_count,
        "labels": "|".join(sorted(set(labels))),
        "shape_types": "|".join(sorted(set(shape_types))),
        "image_sha256": sha256_file(image_path),
        "json_sha256": sha256_file(json_path) if json_path.exists() else "",
        "image_error": image_error,
        "json_error": json_error,
        "scene_type": "pending",
        "boundary_type": "pending",
        "lighting": "pending",
        "curvature": "pending",
        "quality": "pending",
        "difficulty": "pending",
        "review_notes": "",
    }


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scan_source(source_root):
    rows = []
    for source_split in ("train_val", "test"):
        folder = source_root / source_split
        if not folder.exists():
            raise FileNotFoundError(folder)
        images = sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: natural_key(path.name),
        )
        for image_path in images:
            rows.append(
                inspect_sample(
                    image_path,
                    image_path.with_suffix(".json"),
                    source_split,
                )
            )
    return rows


def build_session_template(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["capture_session"]].append(row)

    output = []
    for session, items in sorted(groups.items(), key=lambda item: natural_key(item[0])):
        prefixes = sorted({item["prefix"] for item in items}, key=natural_key)
        output.append(
            {
                "capture_session": session,
                "prefixes": "|".join(prefixes),
                "source_domain": session,
                "source_split": "|".join(sorted({item["source_split"] for item in items})),
                "samples": len(items),
                "positive_samples": sum(int(item["has_boundary"]) for item in items),
                "negative_samples": sum(not int(item["has_boundary"]) for item in items),
                "scene_type": "pending",
                "boundary_type": "pending",
                "lighting": "pending",
                "curvature": "pending",
                "quality": "pending",
                "difficulty": "pending",
                "notes": "",
            }
        )
    return output


def build_continuous_segment_template(rows):
    output = []
    for prefix in sorted(CONTINUOUS_PREFIXES, key=natural_key):
        items = [
            row
            for row in rows
            if row["source_split"] == "train_val" and row["prefix"] == prefix
        ]
        items.sort(key=lambda row: (int(row["frame_order"]), natural_key(row["sample_id"])))
        if not items:
            continue
        output.append(
            {
                "capture_session": "session_021_026",
                "segment_id": f"session_021_026_{prefix}_pending",
                "prefix": prefix,
                "start_sample": items[0]["sample_id"],
                "end_sample": items[-1]["sample_id"],
                "start_frame": items[0]["frame_order"],
                "end_frame": items[-1]["frame_order"],
                "samples": len(items),
                "scene_type": "pending",
                "boundary_type": "pending",
                "lighting": "pending",
                "curvature": "pending",
                "quality": "pending",
                "difficulty": "pending",
                "notes": "Split this row when a substantial scene transition is visible.",
            }
        )
    return output


def fit_thumbnail(image, size):
    image = image.convert("RGB")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def create_contact_sheets(rows, output_dir, step, max_samples, columns, page_rows, thumb_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    page_capacity = columns * page_rows
    label_height = 24
    cell_width = thumb_size[0]
    cell_height = thumb_size[1] + label_height

    for prefix in sorted(CONTINUOUS_PREFIXES, key=natural_key):
        items = [
            row
            for row in rows
            if row["source_split"] == "train_val" and row["prefix"] == prefix
        ]
        items.sort(key=lambda row: (int(row["frame_order"]), natural_key(row["sample_id"])))
        effective_step = (
            max(1, step)
            if step > 0
            else max(1, math.ceil(len(items) / max(1, max_samples)))
        )
        selected_indices = set(range(0, len(items), effective_step))
        if items:
            selected_indices.update({0, len(items) - 1})
        selected = [items[index] for index in sorted(selected_indices)]

        for page_no in range(math.ceil(len(selected) / page_capacity)):
            page_items = selected[page_no * page_capacity : (page_no + 1) * page_capacity]
            canvas = Image.new(
                "RGB",
                (columns * cell_width, page_rows * cell_height),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            page_name = f"{prefix}_page_{page_no + 1:02d}.jpg"
            for slot, row in enumerate(page_items):
                col = slot % columns
                line = slot // columns
                x = col * cell_width
                y = line * cell_height
                with Image.open(row["image_path"]) as image:
                    thumb = fit_thumbnail(image, thumb_size)
                canvas.paste(thumb, (x, y))
                status = "POS" if int(row["has_boundary"]) else "NEG"
                draw.text((x + 4, y + thumb_size[1] + 4), f"{row['sample_id']}  {status}", fill="black")
                index_rows.append(
                    {
                        "prefix": prefix,
                        "page": page_name,
                        "slot": slot + 1,
                        "sample_id": row["sample_id"],
                        "frame_order": row["frame_order"],
                        "has_boundary": row["has_boundary"],
                        "sampling_step": effective_step,
                        "suggested_transition_after": "",
                        "review_notes": "",
                    }
                )
            canvas.save(output_dir / page_name, quality=92)

    write_csv(output_dir / "contact_sheet_index.csv", index_rows)


def build_report(rows, session_rows, output_root):
    source_counts = Counter(row["source_split"] for row in rows)
    prefix_counts = Counter(
        (row["source_split"], row["prefix"]) for row in rows
    )
    positives = Counter(row["source_split"] for row in rows if int(row["has_boundary"]))
    bad_json = [row for row in rows if row["json_error"]]
    bad_images = [row for row in rows if row["image_error"]]
    bad_size = [row for row in rows if (int(row["width"]), int(row["height"])) != (800, 450)]
    json_size_mismatch = [
        row
        for row in rows
        if row["json_width"] != ""
        and row["json_height"] != ""
        and (int(row["json_width"]), int(row["json_height"]))
        != (int(row["width"]), int(row["height"]))
    ]

    lines = [
        "# Emergency Outer Boundary Dataset Inventory",
        "",
        "This inventory is non-destructive. Original images and LabelMe JSON files were not modified.",
        "",
        "## Overall",
        "",
        "| Source | Samples | Positive | Negative |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source in ("train_val", "test"):
        total = source_counts[source]
        positive = positives[source]
        lines.append(f"| {source} | {total} | {positive} | {total - positive} |")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Invalid JSON or missing JSON: {len(bad_json)}",
            f"- Invalid images: {len(bad_images)}",
            f"- Images not 800x450: {len(bad_size)}",
            f"- JSON/image size mismatches: {len(json_size_mismatch)}",
            f"- Capture sessions: {len(session_rows)}",
            "",
            "## Prefix Counts",
            "",
            "| Source | Prefix | Samples |",
            "| --- | --- | ---: |",
        ]
    )
    for (source, prefix), count in sorted(
        prefix_counts.items(),
        key=lambda item: (item[0][0], natural_key(item[0][1])),
    ):
        lines.append(f"| {source} | {prefix} | {count} |")

    lines.extend(
        [
            "",
            "## Next Review",
            "",
            "1. Review the 021-026 contact sheets.",
            "2. Split rows in `scene_segments_021_026_template.csv` at substantial scene transitions.",
            "3. Fill only segment-level semantic fields; per-image annotation is not required.",
            "4. Do not generate train/validation splits until the capture and segment metadata are confirmed.",
        ]
    )
    report_path = output_root / "reports" / "inventory_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    metadata_dir = output_root / "metadata"
    review_dir = output_root / "reports" / "scene_review_021_026"

    rows = scan_source(source_root)
    write_csv(metadata_dir / "sample_inventory.csv", rows)

    sessions = build_session_template(rows)
    write_csv(metadata_dir / "capture_sessions_template.csv", sessions)

    segment_rows = build_continuous_segment_template(rows)
    write_csv(metadata_dir / "scene_segments_021_026_template.csv", segment_rows)

    create_contact_sheets(
        rows,
        review_dir,
        step=args.contact_step,
        max_samples=args.contact_max_samples,
        columns=args.contact_columns,
        page_rows=args.contact_rows,
        thumb_size=(args.thumbnail_width, args.thumbnail_height),
    )
    build_report(rows, sessions, output_root)

    print(f"Samples inventoried: {len(rows)}")
    print(f"Metadata: {metadata_dir}")
    print(f"021-026 review sheets: {review_dir}")
    print(f"Report: {output_root / 'reports' / 'inventory_report.md'}")


if __name__ == "__main__":
    main()
