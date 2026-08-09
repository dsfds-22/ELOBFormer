#!/usr/bin/env python
"""Export raw binary boundary masks from an MMSeg checkpoint."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from mmcv_lite_compat import install_mmcv_lite_ext_stub
from run_mmseg_boundary import (
    install_mmengine_windows_env_guard,
    install_trusted_checkpoint_compat,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw MMSeg predictions as 8-bit binary PNG masks."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-root",
        default="datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-csv", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks that already exist.",
    )
    args = parser.parse_args()

    os.environ["LANE_DETECTION_PROJECT_ROOT"] = str(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    install_mmcv_lite_ext_stub()
    install_mmengine_windows_env_guard()
    install_trusted_checkpoint_compat()

    from mmengine import Config
    from mmseg.apis import inference_model, init_model

    dataset_root = Path(args.dataset_root).resolve()
    split_csv = (
        Path(args.split_csv).resolve()
        if args.split_csv
        else dataset_root / "splits" / f"{args.split}.csv"
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with split_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    config = Config.fromfile(str(Path(args.config).resolve()))
    config.test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(
            type="PackSegInputs",
            meta_keys=(
                "img_path",
                "ori_shape",
                "img_shape",
                "pad_shape",
                "scale_factor",
                "flip",
                "flip_direction",
            ),
        ),
    ]
    model = init_model(
        config,
        str(Path(args.checkpoint).resolve()),
        device=args.device,
    )

    generated = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id") or Path(row["image"]).stem
        output_path = out_dir / f"{sample_id}.png"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image_path = resolve_path(dataset_root, row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"{sample_id}: image not found: {image_path}")

        result = inference_model(model, str(image_path))
        prediction = (
            result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
        )
        prediction = (prediction > 0).astype(np.uint8) * 255

        expected_height = int(row.get("height") or 450)
        expected_width = int(row.get("width") or 800)
        if prediction.shape != (expected_height, expected_width):
            prediction = cv2.resize(
                prediction,
                (expected_width, expected_height),
                interpolation=cv2.INTER_NEAREST,
            )
        if not cv2.imwrite(str(output_path), prediction):
            raise OSError(f"Failed to write prediction: {output_path}")

        generated += 1
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(
                f"Exported {index}/{len(rows)} | "
                f"generated={generated} skipped={skipped}",
                flush=True,
            )

    print(
        f"Saved {generated} masks to {out_dir} "
        f"({skipped} existing masks skipped)."
    )


if __name__ == "__main__":
    main()
