import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image(json_path, image_path_value):
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


def draw_lines(size, polylines, thickness):
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for points in polylines:
        if len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        pts = np.rint(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            mask,
            [pts],
            isClosed=False,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
    return mask


def adaptive_width_at_y(y, height, min_width, max_width, gamma):
    """Perspective-aware line width: thin near horizon, wider near image bottom."""
    if height <= 1:
        return int(max_width)
    ratio = float(np.clip(y / float(height - 1), 0.0, 1.0))
    width = float(min_width) + (float(max_width) - float(min_width)) * (ratio ** float(gamma))
    return int(max(1, round(width)))


def draw_adaptive_lines(size, polylines, min_width, max_width, gamma):
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for points in polylines:
        if len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            y_mid = 0.5 * (float(p0[1]) + float(p1[1]))
            thickness = adaptive_width_at_y(y_mid, height, min_width, max_width, gamma)
            segment = np.rint(np.stack([p0, p1], axis=0)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                mask,
                [segment],
                isClosed=False,
                color=255,
                thickness=thickness,
                lineType=cv2.LINE_8,
            )
        for point in pts:
            thickness = adaptive_width_at_y(float(point[1]), height, min_width, max_width, gamma)
            cv2.circle(
                mask,
                tuple(np.rint(point).astype(np.int32)),
                max(1, thickness // 2),
                255,
                -1,
                lineType=cv2.LINE_8,
            )
    return mask


def make_heatmap(size, polylines, radius, sigma):
    centerline = draw_lines(size, polylines, thickness=1)
    if centerline.max() == 0:
        return np.zeros_like(centerline, dtype=np.uint8)

    inv = np.where(centerline > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    heat = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    heat[dist > radius] = 0.0
    heat = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return heat


def extract_boundary_polylines(data, labels):
    label_set = set(labels)
    polylines = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        shape_type = str(shape.get("shape_type", "")).strip()
        if label not in label_set:
            continue
        if shape_type != "linestrip":
            continue
        points = shape.get("points", [])
        cleaned = []
        for point in points:
            if isinstance(point, list) and len(point) >= 2:
                cleaned.append([float(point[0]), float(point[1])])
        if len(cleaned) >= 2:
            polylines.append(cleaned)
    return polylines


def save_gray(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe boundary polylines to band mask, thin edge mask, and soft heatmap."
    )
    parser.add_argument(
        "--root",
        required=True,
        nargs="+",
        help="One or more input LabelMe dataset roots.",
    )
    parser.add_argument("--out", required=True, help="Output dataset root.")
    parser.add_argument(
        "--boundary-labels",
        nargs="+",
        default=["right_boundary", "emergency_outer_boundary"],
        help="LabelMe labels treated as emergency outer boundary.",
    )
    parser.add_argument(
        "--fixed-band-width",
        type=int,
        default=None,
        help="Use a fixed boundary band thickness. If omitted, perspective-adaptive band width is used.",
    )
    parser.add_argument(
        "--band-min-width",
        type=int,
        default=4,
        help="Perspective-adaptive band width near the far field/top of the image.",
    )
    parser.add_argument(
        "--band-max-width",
        type=int,
        default=12,
        help="Perspective-adaptive band width near the near field/bottom of the image.",
    )
    parser.add_argument(
        "--band-width-gamma",
        type=float,
        default=1.0,
        help="Perspective-adaptive width curve. 1.0 is linear; >1 keeps far-field labels thinner for longer.",
    )
    parser.add_argument("--edge-width", type=int, default=5, help="Thin edge polyline thickness in pixels.")
    parser.add_argument("--heatmap-radius", type=float, default=9.0, help="Gaussian heatmap cutoff radius in pixels.")
    parser.add_argument("--heatmap-sigma", type=float, default=3.0, help="Gaussian heatmap sigma in pixels.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output/images.")
    parser.add_argument("--copy-json", action="store_true", help="Copy original LabelMe JSON into output/labelme_json.")
    parser.add_argument(
        "--split-dir",
        default=None,
        help=(
            "Optional protocol split directory containing train.csv, val.csv, "
            "and test.csv. Matching training manifests are written to output/splits."
        ),
    )
    args = parser.parse_args()

    roots = [Path(value) for value in args.root]
    out = Path(args.out)
    images_dir = out / "images"
    band_dir = out / "masks_band"
    edge_dir = out / "masks_edge"
    heatmap_dir = out / "heatmaps"
    json_dir = out / "labelme_json"
    manifest_path = out / "samples.csv"

    for folder in [images_dir, band_dir, edge_dir, heatmap_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    if args.copy_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    converted = 0
    with_boundary = 0
    missing_image = 0
    bad_json = 0
    seen_stems = set()

    json_paths = sorted(
        (json_path for root in roots for json_path in root.rglob("*.json")),
        key=lambda path: (path.stem, str(path)),
    )
    for json_path in json_paths:
        try:
            data = load_json(json_path)
        except Exception:
            bad_json += 1
            continue

        image_path = find_image(json_path, data.get("imagePath"))
        if image_path is None:
            missing_image += 1
            continue

        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            with Image.open(image_path) as img:
                width, height = img.size

        stem = image_path.stem
        if stem in seen_stems:
            raise ValueError(
                f"Duplicate sample stem '{stem}' found while converting {json_path}. "
                "Sample filenames must be unique across all input roots."
            )
        seen_stems.add(stem)
        image_ext = image_path.suffix.lower()
        image_name = f"{stem}{image_ext}"
        mask_name = f"{stem}.png"

        polylines = extract_boundary_polylines(data, args.boundary_labels)
        has_boundary = len(polylines) > 0
        if has_boundary:
            with_boundary += 1

        if args.fixed_band_width is None:
            band = draw_adaptive_lines(
                (width, height),
                polylines,
                args.band_min_width,
                args.band_max_width,
                args.band_width_gamma,
            )
        else:
            band = draw_lines((width, height), polylines, args.fixed_band_width)
        edge = draw_lines((width, height), polylines, args.edge_width)
        heatmap = make_heatmap((width, height), polylines, args.heatmap_radius, args.heatmap_sigma)

        save_gray(band_dir / mask_name, band)
        save_gray(edge_dir / mask_name, edge)
        save_gray(heatmap_dir / mask_name, heatmap)

        if args.copy_images:
            shutil.copy2(image_path, images_dir / image_name)
        if args.copy_json:
            shutil.copy2(json_path, json_dir / json_path.name)

        rows.append(
            {
                "image": (
                    f"images/{image_name}"
                    if args.copy_images
                    else str(image_path.resolve())
                ),
                "band_mask": f"masks_band/{mask_name}",
                "edge_mask": f"masks_edge/{mask_name}",
                "heatmap": f"heatmaps/{mask_name}",
                "json": (
                    f"labelme_json/{json_path.name}"
                    if args.copy_json
                    else str(json_path.resolve())
                ),
                "has_boundary": int(has_boundary),
                "boundary_polyline_count": len(polylines),
                "width": width,
                "height": height,
            }
        )
        converted += 1

    out.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "image",
            "band_mask",
            "edge_mask",
            "heatmap",
            "json",
            "has_boundary",
            "boundary_polyline_count",
            "width",
            "height",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.split_dir:
        converted_by_id = {
            Path(row["image"]).stem: row
            for row in rows
        }
        output_split_dir = out / "splits"
        output_split_dir.mkdir(parents=True, exist_ok=True)
        assigned_ids = set()

        for split_name in ("train", "val", "test"):
            source_split_path = Path(args.split_dir) / f"{split_name}.csv"
            with source_split_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                source_rows = list(csv.DictReader(handle))

            output_rows = []
            for source_row in source_rows:
                sample_id = source_row["sample_id"]
                if sample_id not in converted_by_id:
                    raise ValueError(
                        f"Split sample '{sample_id}' was not converted"
                    )
                converted_row = converted_by_id[sample_id]
                output_row = dict(source_row)
                output_row["image"] = converted_row["image"]
                output_row["json"] = converted_row["json"]
                output_row["band_mask"] = converted_row["band_mask"]
                output_row["edge_mask"] = converted_row["edge_mask"]
                output_row["heatmap"] = converted_row["heatmap"]
                output_row["has_boundary"] = converted_row["has_boundary"]
                output_row["boundary_polyline_count"] = converted_row[
                    "boundary_polyline_count"
                ]
                output_rows.append(output_row)
                assigned_ids.add(sample_id)

            output_fields = list(output_rows[0])
            with (output_split_dir / f"{split_name}.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=output_fields)
                writer.writeheader()
                writer.writerows(output_rows)

            with (output_split_dir / f"{split_name}.txt").open(
                "w", encoding="utf-8"
            ) as handle:
                for output_row in output_rows:
                    handle.write(f"{output_row['sample_id']}\n")

        converted_ids = set(converted_by_id)
        if assigned_ids != converted_ids:
            missing_from_splits = sorted(converted_ids - assigned_ids)
            unknown_in_splits = sorted(assigned_ids - converted_ids)
            raise ValueError(
                "Converted/split membership mismatch: "
                f"missing_from_splits={missing_from_splits[:10]}, "
                f"unknown_in_splits={unknown_in_splits[:10]}"
            )

    summary = {
        "input_roots": [str(root) for root in roots],
        "output_root": str(out),
        "converted": converted,
        "with_boundary": with_boundary,
        "without_boundary": converted - with_boundary,
        "missing_image": missing_image,
        "bad_json": bad_json,
        "fixed_band_width": args.fixed_band_width,
        "band_min_width": args.band_min_width,
        "band_max_width": args.band_max_width,
        "band_width_gamma": args.band_width_gamma,
        "edge_width": args.edge_width,
        "heatmap_radius": args.heatmap_radius,
        "heatmap_sigma": args.heatmap_sigma,
        "boundary_labels": args.boundary_labels,
    }
    (out / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
