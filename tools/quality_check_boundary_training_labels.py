#!/usr/bin/env python
"""Validate converted emergency-boundary masks and training manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT
        / "datasets"
        / "emergency_outer_boundary_grouped_fixed12_v2_clean601",
    )
    parser.add_argument("--expected-width", type=int, default=800)
    parser.add_argument("--expected-height", type=int, default=450)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gray(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise ValueError(f"Cannot read mask: {path}")
    return array


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    samples = load_csv(root / "samples.csv")

    issues: list[str] = []
    stats = Counter()
    band_pixels: list[int] = []
    edge_pixels: list[int] = []
    sample_ids: set[str] = set()

    for row in samples:
        sample_id = Path(row["image"]).stem
        if sample_id in sample_ids:
            issues.append(f"Duplicate sample ID: {sample_id}")
        sample_ids.add(sample_id)

        band = read_gray(root / row["band_mask"])
        edge = read_gray(root / row["edge_mask"])
        heatmap = read_gray(root / row["heatmap"])
        expected_shape = (args.expected_height, args.expected_width)
        if band.shape != expected_shape:
            issues.append(f"{sample_id}: band shape {band.shape}")
        if edge.shape != expected_shape:
            issues.append(f"{sample_id}: edge shape {edge.shape}")
        if heatmap.shape != expected_shape:
            issues.append(f"{sample_id}: heatmap shape {heatmap.shape}")

        band_values = set(np.unique(band).tolist())
        edge_values = set(np.unique(edge).tolist())
        if not band_values.issubset({0, 255}):
            issues.append(f"{sample_id}: non-binary band values")
        if not edge_values.issubset({0, 255}):
            issues.append(f"{sample_id}: non-binary edge values")

        band_positive = band > 0
        edge_positive = edge > 0
        heat_positive = heatmap > 0
        has_boundary = row["has_boundary"] == "1"

        if has_boundary:
            stats["positive_samples"] += 1
            if not band_positive.any():
                issues.append(f"{sample_id}: positive sample has empty band")
            if not edge_positive.any():
                issues.append(f"{sample_id}: positive sample has empty edge")
            if not heat_positive.any() or heatmap.max() != 255:
                issues.append(f"{sample_id}: invalid positive heatmap")
            band_pixels.append(int(band_positive.sum()))
            edge_pixels.append(int(edge_positive.sum()))
        else:
            stats["negative_samples"] += 1
            if band_positive.any():
                issues.append(f"{sample_id}: negative sample has non-empty band")
            if edge_positive.any():
                issues.append(f"{sample_id}: negative sample has non-empty edge")
            if heat_positive.any():
                issues.append(
                    f"{sample_id}: negative sample has non-empty heatmap"
                )

        if np.logical_and(edge_positive, ~band_positive).any():
            issues.append(f"{sample_id}: edge extends outside band")

    split_counts: dict[str, dict[str, int]] = {}
    assigned_ids: set[str] = set()
    for split in ("train", "val", "test"):
        rows = load_csv(root / "splits" / f"{split}.csv")
        ids = {row["sample_id"] for row in rows}
        if assigned_ids & ids:
            issues.append(f"{split}: sample overlap with another split")
        assigned_ids.update(ids)
        positive = sum(row["has_boundary"] == "1" for row in rows)
        split_counts[split] = {
            "samples": len(rows),
            "positive": positive,
            "negative": len(rows) - positive,
        }

        for row in rows:
            for key in ("image", "json", "band_mask", "edge_mask", "heatmap"):
                value = Path(row[key])
                path = value if value.is_absolute() else root / value
                if not path.is_file():
                    issues.append(
                        f"{split}/{row['sample_id']}: missing {key} {path}"
                    )

    if assigned_ids != sample_ids:
        issues.append(
            "Split/sample membership mismatch: "
            f"unassigned={len(sample_ids - assigned_ids)}, "
            f"unknown={len(assigned_ids - sample_ids)}"
        )

    summary = {
        "root": str(root),
        "samples": len(samples),
        "positive_samples": stats["positive_samples"],
        "negative_samples": stats["negative_samples"],
        "expected_size": [args.expected_width, args.expected_height],
        "band_values": [0, 255],
        "edge_values": [0, 255],
        "positive_band_pixels": {
            "min": min(band_pixels, default=0),
            "mean": float(np.mean(band_pixels)) if band_pixels else 0.0,
            "max": max(band_pixels, default=0),
        },
        "positive_edge_pixels": {
            "min": min(edge_pixels, default=0),
            "mean": float(np.mean(edge_pixels)) if edge_pixels else 0.0,
            "max": max(edge_pixels, default=0),
        },
        "splits": split_counts,
        "issue_count": len(issues),
        "issues": issues,
    }
    (root / "quality_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Boundary Training Label Quality Report",
        "",
        f"- Samples: {summary['samples']}",
        f"- Positive: {summary['positive_samples']}",
        f"- Negative: {summary['negative_samples']}",
        f"- Expected size: {args.expected_width}x{args.expected_height}",
        f"- Issues: {summary['issue_count']}",
        "",
        "| Split | Samples | Positive | Negative |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        item = split_counts[split]
        lines.append(
            f"| {split} | {item['samples']} | "
            f"{item['positive']} | {item['negative']} |"
        )
    lines.extend(
        [
            "",
            "## Label Checks",
            "",
            "- Band masks are binary 0/255 images.",
            "- Edge masks are binary 0/255 images.",
            "- Every edge mask is contained inside its band mask.",
            "- Positive samples have non-empty band, edge, and heatmap labels.",
            "- Negative samples have empty band, edge, and heatmap labels.",
            "- Every converted sample appears in exactly one split.",
        ]
    )
    if issues:
        lines.extend(["", "## Issues", "", *[f"- {item}" for item in issues]])
    (root / "quality_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
