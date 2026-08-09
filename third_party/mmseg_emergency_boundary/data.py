"""CSV dataset and sampling components for emergency-boundary training."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional, Sequence

import mmcv
import mmengine.fileio as fileio
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.registry import DATA_SAMPLERS
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS
from PIL import Image, ImageEnhance
from torch.utils.data import Sampler


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@DATASETS.register_module()
class EmergencyBoundaryCsvDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "emergency_outer_boundary"),
        palette=[[0, 0, 0], [0, 255, 0]],
    )

    def load_data_list(self) -> list[dict]:
        csv_path = Path(self.ann_file)
        root = Path(self.data_root)
        data_list = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))

        group_fields = (
            "reviewed_segment_id",
            "capture_session",
            "prefix",
        )
        for row in rows:
            image_path = _resolve_path(root, row["image"])
            band_path = _resolve_path(root, row["band_mask"])
            edge_value = row.get("edge_mask", "") or row["band_mask"]
            edge_path = _resolve_path(root, edge_value)
            missing = [
                str(path)
                for path in (image_path, band_path, edge_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{row.get('sample_id', '<unknown>')}: missing files "
                    + ", ".join(missing)
                )

            scene_group = ""
            for field in group_fields:
                value = str(row.get(field, "")).strip()
                if value:
                    scene_group = f"{field}:{value}"
                    break
            if not scene_group:
                scene_group = f"sample:{row['sample_id']}"

            data_list.append(
                dict(
                    img_path=str(image_path),
                    seg_map_path=str(band_path),
                    edge_map_path=str(edge_path),
                    sample_id=row["sample_id"],
                    prefix=str(row.get("prefix", "")),
                    capture_session=str(row.get("capture_session", "")),
                    reviewed_segment_id=str(
                        row.get("reviewed_segment_id", "")
                    ),
                    scene_group=scene_group,
                    has_boundary=int(row.get("has_boundary", 0)),
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                )
            )
        return data_list


@TRANSFORMS.register_module()
class LoadEmergencyBoundaryAnnotations(BaseTransform):
    def __init__(
        self,
        expected_size: Sequence[int] = (450, 800),
        backend_args: Optional[dict] = None,
    ) -> None:
        self.expected_size = tuple(int(value) for value in expected_size)
        self.backend_args = backend_args

    def _load_binary(self, path: str) -> np.ndarray:
        content = fileio.get(path, backend_args=self.backend_args)
        image = mmcv.imfrombytes(
            content, flag="grayscale", backend="pillow"
        )
        return (image > 0).astype(np.uint8)

    def transform(self, results: dict) -> dict:
        band = self._load_binary(results["seg_map_path"])
        edge = self._load_binary(results["edge_map_path"])
        if band.shape != self.expected_size or edge.shape != self.expected_size:
            raise ValueError(
                f"{results['sample_id']}: expected masks "
                f"{self.expected_size}, got band={band.shape}, edge={edge.shape}"
            )
        if tuple(results["img"].shape[:2]) != self.expected_size:
            raise ValueError(
                f"{results['sample_id']}: expected image "
                f"{self.expected_size}, got {results['img'].shape[:2]}"
            )
        results["gt_seg_map"] = band
        results["gt_edge_map"] = edge
        results["seg_fields"].append("gt_seg_map")
        return results


@TRANSFORMS.register_module()
class EmergencyBoundaryBasicAugmentation(BaseTransform):
    """Match the frozen SegFormer baseline's image-only augmentation."""

    def transform(self, results: dict) -> dict:
        rgb = np.ascontiguousarray(results["img"][..., ::-1])
        image = Image.fromarray(rgb)
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.75, 1.25)
            )
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.75, 1.25)
            )
        if random.random() < 0.3:
            image = ImageEnhance.Color(image).enhance(
                random.uniform(0.85, 1.15)
            )
        results["img"] = np.ascontiguousarray(np.asarray(image)[..., ::-1])
        return results


@DATA_SAMPLERS.register_module()
class SqrtSceneSampler(Sampler[int]):
    """Sample each item with weight 1/sqrt(scene size)."""

    def __init__(
        self,
        dataset,
        replacement: bool = True,
        num_samples: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        dataset.full_init()
        self.dataset = dataset
        self.replacement = bool(replacement)
        self.num_samples = int(num_samples) or len(dataset)
        self.seed = int(seed or 0)
        self.epoch = 0

        groups = [
            str(dataset.get_data_info(index)["scene_group"])
            for index in range(len(dataset))
        ]
        counts = Counter(groups)
        self.weights = torch.tensor(
            [1.0 / math.sqrt(counts[group]) for group in groups],
            dtype=torch.double,
        )
        self.group_count = len(counts)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
