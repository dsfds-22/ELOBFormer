#!/usr/bin/env python
"""Create leakage-safe train/val/test manifests by complete capture session."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "datasets" / "emergency_outer_boundary_protocol_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split train_val samples by complete capture sessions and preserve "
            "the independent source test set."
        )
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_PROTOCOL / "metadata" / "sample_inventory_segmented.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PROTOCOL / "splits",
    )
    parser.add_argument(
        "--val-sessions",
        nargs="+",
        required=True,
        help="Complete capture_session values assigned to validation.",
    )
    parser.add_argument("--target-val-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Inventory is empty: {path}")
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    positive = sum(row["has_boundary"] == "1" for row in rows)
    return {
        "samples": len(rows),
        "positive": positive,
        "negative": len(rows) - positive,
        "prefixes": sorted({row["prefix"] for row in rows}),
        "capture_sessions": sorted({row["capture_session"] for row in rows}),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "sample_id",
        "prefix",
        "capture_session",
        "scene_segment",
        "reviewed_segment_id",
        "split",
        "image",
        "json",
        "has_boundary",
        "boundary_type",
        "shoulder_state",
        "occlusion",
        "annotation_scope",
        "difficulty",
        "image_sha256",
        "json_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(f"{row['sample_id']}\n" for row in rows),
        encoding="utf-8",
    )


def output_row(row: dict[str, str], split: str) -> dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "prefix": row["prefix"],
        "capture_session": row["capture_session"],
        "scene_segment": row["scene_segment"],
        "reviewed_segment_id": row.get("reviewed_segment_id", ""),
        "split": split,
        "image": row["image_path"],
        "json": row["json_path"],
        "has_boundary": row["has_boundary"],
        "boundary_type": row.get("reviewed_boundary_type")
        or row.get("boundary_type", ""),
        "shoulder_state": row.get("reviewed_shoulder_state", ""),
        "occlusion": row.get("reviewed_occlusion", ""),
        "annotation_scope": row.get("reviewed_annotation_scope", ""),
        "difficulty": row.get("reviewed_difficulty")
        or row.get("difficulty", ""),
        "image_sha256": row["image_sha256"],
        "json_sha256": row["json_sha256"],
    }


def check_no_leakage(split_rows: dict[str, list[dict[str, str]]]) -> None:
    session_splits: dict[str, set[str]] = defaultdict(set)
    sample_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)

    for split, rows in split_rows.items():
        for row in rows:
            session_splits[row["capture_session"]].add(split)
            sample_splits[row["sample_id"]].add(split)
            hash_splits[row["image_sha256"]].add(split)

    session_leaks = {
        key: value for key, value in session_splits.items() if len(value) > 1
    }
    sample_leaks = {
        key: value for key, value in sample_splits.items() if len(value) > 1
    }
    hash_leaks = {
        key: value for key, value in hash_splits.items() if len(value) > 1
    }
    if session_leaks or sample_leaks or hash_leaks:
        raise ValueError(
            "Split leakage detected: "
            f"sessions={len(session_leaks)}, "
            f"sample_ids={len(sample_leaks)}, "
            f"image_hashes={len(hash_leaks)}"
        )


def build_report(
    split_rows: dict[str, list[dict[str, str]]],
    target_val_ratio: float,
) -> str:
    stats = {split: summarize(rows) for split, rows in split_rows.items()}
    development_total = stats["train"]["samples"] + stats["val"]["samples"]
    actual_val_ratio = stats["val"]["samples"] / development_total

    lines = [
        "# Capture-Group Split Report",
        "",
        "The train/validation split is isolated by complete capture session.",
        "The independent source `test` folder is preserved unchanged.",
        "",
        "## Overall",
        "",
        "| Split | Samples | Development ratio | Positive | Negative | Prefixes | Sessions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        item = stats[split]
        ratio = (
            f"{item['samples'] / development_total:.2%}"
            if split != "test"
            else "independent"
        )
        lines.append(
            f"| {split} | {item['samples']} | {ratio} | "
            f"{item['positive']} | {item['negative']} | "
            f"{len(item['prefixes'])} | {len(item['capture_sessions'])} |"
        )

    lines.extend(
        [
            "",
            f"- Target validation ratio: {target_val_ratio:.2%}",
            f"- Actual validation ratio: {actual_val_ratio:.2%}",
            f"- Validation sessions: {', '.join(stats['val']['capture_sessions'])}",
            "",
            "## Per-prefix Allocation",
            "",
            "| Split | Prefix | Samples | Positive | Negative |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split in ("train", "val", "test"):
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in split_rows[split]:
            grouped[row["prefix"]]["samples"] += 1
            grouped[row["prefix"]]["positive"] += int(
                row["has_boundary"] == "1"
            )
        for prefix in sorted(grouped):
            item = grouped[prefix]
            lines.append(
                f"| {split} | {prefix} | {item['samples']} | "
                f"{item['positive']} | "
                f"{item['samples'] - item['positive']} |"
            )

    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            "- Capture session overlap across splits: 0",
            "- Sample ID overlap across splits: 0",
            "- Exact image hash overlap across splits: 0",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_val_ratio < 1.0:
        raise ValueError("--target-val-ratio must be between 0 and 1")

    rows = load_inventory(args.inventory)
    development_sessions = {
        row["capture_session"]
        for row in rows
        if row["source_split"] == "train_val"
    }
    requested_val = set(args.val_sessions)
    unknown = sorted(requested_val - development_sessions)
    if unknown:
        raise ValueError(f"Unknown train_val capture sessions: {unknown}")
    if not requested_val:
        raise ValueError("At least one validation session is required")

    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for row in rows:
        if row["source_split"] == "test":
            split = "test"
        elif row["capture_session"] in requested_val:
            split = "val"
        else:
            split = "train"
        split_rows[split].append(output_row(row, split))

    for rows_for_split in split_rows.values():
        rows_for_split.sort(
            key=lambda item: (
                item["prefix"],
                item["sample_id"],
            )
        )

    check_no_leakage(split_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows_for_split in split_rows.items():
        write_csv(args.out_dir / f"{split}.csv", rows_for_split)
        write_txt(args.out_dir / f"{split}.txt", rows_for_split)

    combined = [
        row
        for split in ("train", "val", "test")
        for row in split_rows[split]
    ]
    write_csv(args.out_dir / "all_samples.csv", combined)

    summary = {
        "strategy": "complete_capture_session_holdout",
        "inventory": str(args.inventory.resolve()),
        "target_val_ratio": args.target_val_ratio,
        "validation_sessions": sorted(requested_val),
        **{
            split: summarize(rows_for_split)
            for split, rows_for_split in split_rows.items()
        },
    }
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "split_report.md").write_text(
        build_report(split_rows, args.target_val_ratio),
        encoding="utf-8",
    )

    print(f"Wrote grouped split manifests: {args.out_dir}")
    for split in ("train", "val", "test"):
        item = summary[split]
        print(
            f"{split}: {item['samples']} samples, "
            f"{item['positive']} positive, {item['negative']} negative"
        )


if __name__ == "__main__":
    main()
