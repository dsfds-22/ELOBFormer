"""Create a paper-ready visualization of the three supervision labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BOUNDARY_LABELS = {"right_boundary", "emergency_outer_boundary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601"),
    )
    parser.add_argument("--sample-id", default="021_0255_rgb")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "reports/paper_figures/chapter3_three_level_supervision/021_0255_rgb"
        ),
    )
    return parser.parse_args()


def find_sample(dataset_root: Path, sample_id: str) -> tuple[str, dict[str, str]]:
    for split in ("train", "val", "test"):
        csv_path = dataset_root / "splits" / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row_id = Path(row.get("image", "")).stem
                if row_id == sample_id:
                    return split, row
    raise FileNotFoundError(f"Sample not found in split files: {sample_id}")


def resolve_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_labelme_overlay(image_bgr: np.ndarray, json_path: Path) -> tuple[np.ndarray, int, int]:
    with json_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    overlay = image_bgr.copy()
    shape_count = 0
    point_count = 0
    for shape in annotation.get("shapes", []):
        if shape.get("label") not in BOUNDARY_LABELS:
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(points) < 2:
            continue
        points = np.rint(points).astype(np.int32)
        cv2.polylines(overlay, [points], False, (0, 0, 255), 4, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), 1, cv2.LINE_AA)
        shape_count += 1
        point_count += len(points)
    if shape_count == 0:
        raise ValueError(f"No boundary polyline found in {json_path}")
    return overlay, shape_count, point_count


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    colored[heatmap == 0] = 0
    return colored


def to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def add_caption(panel: Image.Image, caption: str) -> Image.Image:
    caption_height = 64
    canvas = Image.new("RGB", (panel.width, panel.height + caption_height), "white")
    canvas.paste(panel, (0, caption_height))
    draw = ImageDraw.Draw(canvas)
    font = load_font(30)
    box = draw.textbbox((0, 0), caption, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    draw.text(
        ((panel.width - text_width) / 2, (caption_height - text_height) / 2 - box[1]),
        caption,
        fill="black",
        font=font,
    )
    return canvas


def save_composite(
    panels: list[tuple[str, Image.Image]], png_path: Path, tiff_path: Path
) -> tuple[int, int]:
    captioned = [add_caption(panel, caption) for caption, panel in panels]
    gap = 12
    width = sum(panel.width for panel in captioned) + gap * (len(captioned) - 1)
    height = max(panel.height for panel in captioned)
    canvas = Image.new("RGB", (width, height), (235, 235, 235))
    x = 0
    for panel in captioned:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    canvas.save(png_path, compress_level=3)
    canvas.save(tiff_path, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    return canvas.size


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split, row = find_sample(dataset_root, args.sample_id)
    image_path = resolve_path(dataset_root, row["image"])
    json_path = resolve_path(dataset_root, row["json"])
    band_path = resolve_path(dataset_root, row["band_mask"])
    edge_path = resolve_path(dataset_root, row["edge_mask"])
    heatmap_path = resolve_path(dataset_root, row["heatmap"])

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    band = cv2.imread(str(band_path), cv2.IMREAD_GRAYSCALE)
    edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
    heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or band is None or edge is None or heatmap is None:
        raise FileNotFoundError("One or more source images could not be loaded")
    if image_bgr.shape[:2] != (450, 800):
        raise ValueError(f"Expected 800x450 RGB image, got {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    if band.shape != (450, 800) or edge.shape != (450, 800) or heatmap.shape != (450, 800):
        raise ValueError("All supervision labels must be 800x450")

    overlay, shape_count, point_count = draw_labelme_overlay(image_bgr, json_path)
    heatmap_color = colorize_heatmap(heatmap)

    original_out = out_dir / "01_original_rgb.jpg"
    json_out = out_dir / "02_labelme_annotation.json"
    band_out = out_dir / "03_band_mask_fixed12.png"
    edge_out = out_dir / "04_edge_mask_fixed5.png"
    heatmap_out = out_dir / "05_heatmap_radius9_sigma3.png"
    shutil.copy2(image_path, original_out)
    shutil.copy2(json_path, json_out)
    shutil.copy2(band_path, band_out)
    shutil.copy2(edge_path, edge_out)
    shutil.copy2(heatmap_path, heatmap_out)

    cv2.imwrite(str(out_dir / "02_labelme_polyline_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / "05_heatmap_color.png"), heatmap_color)

    panels = [
        ("(a) Original RGB", to_pil_rgb(image_bgr)),
        ("(b) LabelMe Polyline", to_pil_rgb(overlay)),
        ("(c) 12-px Band", Image.fromarray(band).convert("RGB")),
        ("(d) 5-px Thin Edge", Image.fromarray(edge).convert("RGB")),
        ("(e) Soft Heatmap", to_pil_rgb(heatmap_color)),
    ]
    figure_path = out_dir / "three_level_supervision_figure.png"
    figure_tiff_path = out_dir / "three_level_supervision_figure_600dpi.tif"
    figure_width, figure_height = save_composite(
        panels, figure_path, figure_tiff_path
    )

    readme = f"""# Three-level Supervision Example

- Sample: `{args.sample_id}`
- Split: `{split}`
- Resolution: `800 x 450`
- Boundary annotations: `{shape_count}` polyline(s), `{point_count}` control points
- Band label: fixed total width `12 px`
- Thin-edge label: fixed total width `5 px`
- Soft heatmap: truncated Gaussian, radius `9 px`, sigma `3 px`
- Paper figure: `{figure_width} x {figure_height} px`, lossless LZW TIFF, `600 DPI`

## Source files

- RGB: `{image_path}`
- LabelMe JSON: `{json_path}`
- Band: `{band_path}`
- Edge: `{edge_path}`
- Heatmap: `{heatmap_path}`

The copied grayscale Band, Edge, and Heatmap files are the exact training labels.
`02_labelme_polyline_overlay.png` and `05_heatmap_color.png` are visualization-only files.
`three_level_supervision_figure.png` is a convenient preview.
`three_level_supervision_figure_600dpi.tif` is the paper-ready five-panel figure.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved paper figure and source materials to: {out_dir}")


if __name__ == "__main__":
    main()
