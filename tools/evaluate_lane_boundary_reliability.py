import argparse
import contextlib
import csv
import io
import importlib.machinery
import json
import math
import os
import re
import sys
import time
import types
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable=None, *args, **kwargs: iterable
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.__spec__ = importlib.machinery.ModuleSpec(
        "tqdm", loader=None
    )
    tqdm_stub.tqdm = tqdm
    sys.modules["tqdm"] = tqdm_stub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
UFLD_ROOT = PROJECT_ROOT / "third_party" / "ufld"
if str(UFLD_ROOT) not in sys.path:
    sys.path.insert(0, str(UFLD_ROOT))

from data.constant import culane_row_anchor
from tools.auto_label_lanes_to_labelme import (
    build_model as build_lane_model,
    predict_lanes,
)
from tools.evaluate_raw_boundary_metrics import boundary_metrics
from tools.v4_presence_verifier import load_v4_presence_verifier
from train_emergency_boundary_baseline import (
    BoundaryDataset,
    SegFormerBandOnlyDeployment,
    band_logits,
    build_model_from_config,
)


LANE_LABELS = ("lane1", "lane2", "lane3", "lane4")
LANE_COLORS = {
    "lane1": (255, 215, 0),
    "lane2": (255, 0, 255),
    "lane3": (255, 140, 0),
    "lane4": (0, 220, 255),
}
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate emergency outer-boundary predictions using ten lower-image "
            "lane-distance intervals and gate the final right-boundary output."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "emergency_boundary_segformer_b1_sqrt_scene_band_edge03_heatmap01.yaml"
        ),
    )
    parser.add_argument(
        "--boundary-backend",
        choices=["native", "mmseg"],
        default="native",
        help="Boundary checkpoint implementation.",
    )
    parser.add_argument(
        "--deploy-band-only",
        action="store_true",
        help="Remove training-only Edge/Heatmap heads during native inference.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--lane-sources",
        nargs="+",
        choices=["labelme", "model"],
        default=["labelme", "model"],
    )
    parser.add_argument(
        "--lane-model",
        default="lane_finetune_labelme_ep041.pth",
        help="Existing project-specific CULane lane checkpoint.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--band-threshold", type=float, default=0.5)
    parser.add_argument("--roi-start", type=float, default=0.50)
    parser.add_argument("--roi-end", type=float, default=0.90)
    parser.add_argument("--num-intervals", type=int, default=10)
    parser.add_argument(
        "--sampling-mode",
        choices=["uniform", "perspective"],
        default="uniform",
        help=(
            "uniform preserves the frozen V2 interval layout; perspective "
            "uses bottom-dense non-uniform sample rows for the V3 candidate."
        ),
    )
    parser.add_argument(
        "--perspective-gamma",
        type=float,
        default=1.35,
        help="Bottom-density exponent used only by perspective sampling.",
    )
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--min-valid-rows", type=int, default=3)
    parser.add_argument("--ratio-min", type=float, default=0.40)
    parser.add_argument("--ratio-max", type=float, default=2.0)
    parser.add_argument("--ratio-high-decay", type=float, default=0.5)
    parser.add_argument("--stable-cv", type=float, default=0.20)
    parser.add_argument("--cv-decay", type=float, default=0.30)
    parser.add_argument(
        "--gate-mode",
        choices=["reliable", "reliable_or_uncertain", "off"],
        default="reliable",
        help=(
            "Output-gate policy. The default only outputs predictions whose "
            "V2 status is reliable."
        ),
    )
    parser.add_argument(
        "--gate-lane-source",
        choices=["labelme", "model"],
        default="model",
        help="Lane source used for the formal system-level gate metrics.",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=5.0,
        help="Pixel tolerance used for gated Boundary F1.",
    )
    parser.add_argument(
        "--presence-verifier",
        default="",
        help=(
            "Optional V4 semantic-presence checkpoint. When supplied, the "
            "final output requires both the V2 geometry gate and V4 presence."
        ),
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=-1.0,
        help=(
            "Override the validation-calibrated V4 threshold. A negative "
            "value uses the threshold stored in the V4 checkpoint."
        ),
    )
    parser.add_argument(
        "--presence-smoothing",
        choices=["none", "median"],
        default="none",
        help="Optional causal smoothing for consecutive V4 presence scores.",
    )
    parser.add_argument(
        "--presence-window",
        type=int,
        default=3,
        help="Causal median window length.",
    )
    parser.add_argument(
        "--presence-max-frame-gap",
        type=int,
        default=5,
        help="Reset temporal state when the numeric frame gap is larger.",
    )
    parser.add_argument(
        "--skip-gated-masks",
        action="store_true",
        help="Do not save the final gated right-boundary masks.",
    )
    parser.add_argument(
        "--save-raw-boundary-masks",
        action="store_true",
        help=(
            "Save the thresholded boundary masks exactly as supplied to the "
            "reliability gate. This is an evidence export and does not alter gating."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=[],
        help="Optional sample-prefix filter, for example: --prefixes 049.",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Write CSV/JSON reports without per-image visualizations.",
    )
    parser.add_argument(
        "--export-hard-negative-csv",
        default="",
        help=(
            "Optional output CSV for V2-guided hard-negative fine-tuning. "
            "Only model-lane reliable/uncertain predictions on GT-negative "
            "samples receive an extra mask and weight."
        ),
    )
    parser.add_argument(
        "--hard-negative-mask-dir",
        default="",
        help="Mask directory used with --export-hard-negative-csv.",
    )
    parser.add_argument(
        "--hard-negative-min-score",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hard-negative-low-weight",
        type=float,
        default=0.0,
        help="Optional fixed mining weight for low-reliability negative predictions.",
    )
    parser.add_argument(
        "--hard-negative-unavailable-weight",
        type=float,
        default=0.0,
        help=(
            "Optional fixed mining weight for non-empty negative predictions "
            "whose lane geometry is unavailable."
        ),
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_boundary_model(cfg, checkpoint_path, device, deploy_band_only=False):
    model = build_model_from_config(cfg).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    if deploy_band_only:
        model = SegFormerBandOnlyDeployment(model).to(device)
    model.eval()
    return model


class MMSegBoundaryAdapter(torch.nn.Module):
    """Expose an MMSeg two-class model as a native one-logit band model."""

    def __init__(self, model, internal_size):
        super().__init__()
        self.model = model
        self.internal_size = tuple(int(value) for value in internal_size)

    def forward(self, images):
        original_height, original_width = images.shape[-2:]
        pad_height = self.internal_size[0] - original_height
        pad_width = self.internal_size[1] - original_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                f"MMSeg internal size {self.internal_size} is smaller than "
                f"input {(original_height, original_width)}."
            )
        if pad_height or pad_width:
            images = torch.nn.functional.pad(
                images, (0, pad_width, 0, pad_height), value=0.0
            )
        metadata = [
            {
                "img_shape": self.internal_size,
                "ori_shape": (original_height, original_width),
                "pad_shape": self.internal_size,
            }
            for _ in range(images.shape[0])
        ]
        logits = self.model.encode_decode(images, metadata)
        logits = logits[..., :original_height, :original_width]
        foreground_logit = logits[:, 1:2] - logits[:, 0:1]
        return {"band": foreground_logit}


def load_mmseg_boundary_model(config_path, checkpoint_path, device):
    mmseg_root = PROJECT_ROOT / "mmsegmentation-main"
    os.environ["LANE_DETECTION_PROJECT_ROOT"] = str(PROJECT_ROOT)
    if str(mmseg_root) not in sys.path:
        sys.path.insert(0, str(mmseg_root))

    from tools.mmcv_lite_compat import install_mmcv_lite_ext_stub
    from tools.run_mmseg_boundary import install_trusted_checkpoint_compat

    install_mmcv_lite_ext_stub()
    install_trusted_checkpoint_compat()

    from mmengine import Config
    from mmseg.apis import init_model

    config = Config.fromfile(str(config_path.resolve()))
    model = init_model(
        config,
        str(checkpoint_path.resolve()),
        device=str(device),
    )
    preprocessor_cfg = config.model.get(
        "data_preprocessor", config.get("data_preprocessor", {})
    )
    internal_size = preprocessor_cfg.get("size", (450, 800))
    return MMSegBoundaryAdapter(model, internal_size).to(device).eval()


def load_project_lane_model(model_path, device):
    lane_args = SimpleNamespace(
        model=str(model_path),
        dataset="CULane",
        backbone="18",
        griding_num=200,
        num_lanes=4,
        min_points=3,
        edge_margin=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model, cls_num_per_lane = build_lane_model(lane_args, device)
    return model, cls_num_per_lane, lane_args


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * IMAGE_STD + IMAGE_MEAN, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def load_labelme_lanes(json_path):
    with Path(json_path).open("r", encoding="utf-8-sig") as stream:
        data = json.load(stream)
    lanes = defaultdict(list)
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if label not in LANE_LABELS:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        lanes[label].append([[float(x), float(y)] for x, y in points])
    return dict(lanes)


def predicted_lanes_as_shapes(model, image_rgb, lane_args, cls_num_per_lane, device):
    pil_image = Image.fromarray(image_rgb)
    lanes = predict_lanes(
        model,
        pil_image,
        lane_args,
        cls_num_per_lane,
        culane_row_anchor,
        device,
    )
    return {label: [points] for label, points in lanes}


def rasterize_lanes(lanes, height, width):
    masks = {}
    for label in LANE_LABELS:
        mask = np.zeros((height, width), dtype=np.uint8)
        for points in lanes.get(label, []):
            array = np.asarray(points, dtype=np.float32)
            if len(array) < 2:
                continue
            array[:, 0] = np.clip(array[:, 0], 0, width - 1)
            array[:, 1] = np.clip(array[:, 1], 0, height - 1)
            cv2.polylines(
                mask,
                [np.rint(array).astype(np.int32)],
                isClosed=False,
                color=1,
                thickness=1,
                lineType=cv2.LINE_8,
            )
        masks[label] = mask
    return masks


def interval_centers(
    height,
    roi_start,
    roi_end,
    num_intervals,
    sampling_mode="uniform",
    perspective_gamma=1.35,
):
    y_start = int(np.clip(round(height * roi_start), 0, height - 1))
    y_end = int(np.clip(round(height * roi_end), y_start + 1, height - 1))
    if sampling_mode == "perspective":
        if perspective_gamma <= 0:
            raise ValueError("perspective_gamma must be positive")
        bottom_up = np.linspace(0.0, 1.0, num_intervals)
        centers = np.rint(
            y_end
            - (y_end - y_start)
            * np.power(bottom_up, perspective_gamma)
        ).astype(int)[::-1]
        centers = np.maximum.accumulate(centers)
        edges = np.empty(num_intervals + 1, dtype=int)
        edges[0] = y_start
        edges[-1] = min(height, y_end + 1)
        if num_intervals > 1:
            edges[1:-1] = np.rint(
                (centers[:-1] + centers[1:]) / 2.0
            ).astype(int)
    else:
        edges = np.rint(
            np.linspace(y_start, y_end, num_intervals + 1)
        ).astype(int)
        centers = np.asarray(
            [
                int(round((edges[i] + edges[i + 1]) / 2.0))
                for i in range(num_intervals)
            ],
            dtype=int,
        )
    return y_start, y_end, edges.tolist(), centers.tolist()


def window_rows(center_y, radius, height):
    start = max(0, center_y - radius)
    end = min(height, center_y + radius + 1)
    return np.arange(start, end, dtype=np.int32)


def mask_window_x(mask, rows, min_valid_rows):
    valid_rows = [int(y) for y in rows if np.any(mask[int(y)] > 0)]
    if len(valid_rows) < min_valid_rows:
        return None, len(valid_rows)
    yx = np.where(mask[valid_rows] > 0)
    if yx[1].size == 0:
        return None, len(valid_rows)
    return float(np.median(yx[1])), len(valid_rows)


def count_window_components(mask, rows):
    window = (mask[rows] > 0).astype(np.uint8)
    if not np.any(window):
        return 0
    count, _ = cv2.connectedComponents(window, connectivity=8)
    return int(max(0, count - 1))


def distance_score(ratio, ratio_min, ratio_max, high_decay):
    if not np.isfinite(ratio):
        return 0.0
    if ratio < ratio_min:
        return float(np.clip(ratio / ratio_min, 0.0, 1.0))
    if ratio <= ratio_max:
        return 1.0
    return float(math.exp(-(ratio - ratio_max) / high_decay))


def interval_distance_class(ratio, ratio_min, ratio_max):
    if ratio is None or not np.isfinite(ratio):
        return "N/A"
    if ratio < ratio_min:
        return "CLOSE"
    if ratio > ratio_max:
        return "FAR"
    return "PASS"


def interval_vertical_weight(interval_index, num_intervals):
    """Weight lower image intervals more strongly than distant upper intervals."""
    if num_intervals <= 1:
        return 1.0
    position = (float(interval_index) - 1.0) / (float(num_intervals) - 1.0)
    return 0.5 + 0.5 * position


def score_intervals(interval_rows, args, has_prediction):
    valid = [row for row in interval_rows if row["valid"]]
    valid_count = len(valid)

    empty = {
        "valid_interval_count": valid_count,
        "aggregate_ratio": None,
        "distance_score": None,
        "ratio_cv": None,
        "stability_score": None,
        "coverage_score": None,
        "distance_pass_rate": None,
        "base_reliability_score": None,
        "reliability_score": None,
    }
    if not has_prediction:
        return {**empty, "status": "no_prediction"}
    if valid_count < 3:
        return {**empty, "status": "unavailable"}

    ratios = np.asarray([row["boundary_ratio"] for row in valid], dtype=np.float64)
    if valid_count >= 6:
        ordered = np.sort(ratios)
        scoring_ratios = ordered[1:-1]
        aggregate_ratio = float(scoring_ratios.mean())
        coverage_score = 1.0
    else:
        scoring_ratios = ratios
        aggregate_ratio = float(np.median(ratios))
        coverage_score = {5: 0.8, 4: 0.6, 3: 0.4}[valid_count]

    mean_ratio = float(scoring_ratios.mean())
    if abs(mean_ratio) <= 1e-9:
        ratio_cv = float("inf")
        stability_score = 0.0
    else:
        ratio_cv = float(scoring_ratios.std(ddof=0) / abs(mean_ratio))
        if ratio_cv <= args.stable_cv:
            stability_score = 1.0
        else:
            stability_score = float(
                math.exp(-(ratio_cv - args.stable_cv) / args.cv_decay)
            )

    dist_score = distance_score(
        aggregate_ratio,
        args.ratio_min,
        args.ratio_max,
        args.ratio_high_decay,
    )
    pass_weight = sum(
        row["vertical_weight"]
        for row in valid
        if row["distance_class"] == "PASS"
    )
    total_weight = sum(row["vertical_weight"] for row in valid)
    distance_pass_rate = pass_weight / total_weight if total_weight > 0 else 0.0

    base_reliability = (
        0.5 * dist_score + 0.3 * stability_score + 0.2 * coverage_score
    )
    reliability = base_reliability * (0.5 + 0.5 * distance_pass_rate)
    if reliability >= 0.75:
        status = "reliable"
    elif reliability >= 0.50:
        status = "uncertain"
    else:
        status = "low_reliability"

    return {
        "valid_interval_count": valid_count,
        "aggregate_ratio": aggregate_ratio,
        "distance_score": dist_score,
        "ratio_cv": ratio_cv,
        "stability_score": stability_score,
        "coverage_score": coverage_score,
        "distance_pass_rate": float(distance_pass_rate),
        "base_reliability_score": float(base_reliability),
        "reliability_score": float(reliability),
        "status": status,
    }


def gate_accepts(status, mode):
    if mode == "off":
        return status != "no_prediction"
    if mode == "reliable_or_uncertain":
        return status in {"reliable", "uncertain"}
    return status == "reliable"


def evaluate_sample(boundary_mask, lane_masks, args):
    height, width = boundary_mask.shape
    _, _, edges, centers = interval_centers(
        height,
        args.roi_start,
        args.roi_end,
        args.num_intervals,
        args.sampling_mode,
        args.perspective_gamma,
    )
    has_prediction = bool(np.any(boundary_mask > 0))
    interval_rows = []

    for interval_index, center_y in enumerate(centers, start=1):
        rows = window_rows(center_y, args.window_radius, height)
        boundary_x, boundary_valid_rows = mask_window_x(
            boundary_mask,
            rows,
            args.min_valid_rows,
        )
        component_count = count_window_components(boundary_mask, rows)

        lane_x = {}
        lane_valid_rows = {}
        for label in LANE_LABELS:
            x_value, valid_row_count = mask_window_x(
                lane_masks[label],
                rows,
                args.min_valid_rows,
            )
            lane_x[label] = x_value
            lane_valid_rows[label] = valid_row_count

        adjacent_widths = []
        for left_label, right_label in zip(LANE_LABELS[:-1], LANE_LABELS[1:]):
            left_x = lane_x[left_label]
            right_x = lane_x[right_label]
            if left_x is None or right_x is None:
                continue
            gap = right_x - left_x
            if gap > 0:
                adjacent_widths.append(float(gap))

        valid_lane_x = [value for value in lane_x.values() if value is not None]
        invalid_reason = ""
        lane_width = None
        rightmost_lane_x = None
        boundary_distance = None
        boundary_ratio = None

        if boundary_x is None:
            invalid_reason = "boundary_missing"
        elif not adjacent_widths:
            invalid_reason = "lane_scale_insufficient"
        elif not valid_lane_x:
            invalid_reason = "lane_missing"
        else:
            lane_width = float(np.mean(adjacent_widths))
            rightmost_lane_x = float(max(valid_lane_x))
            boundary_distance = float(boundary_x - rightmost_lane_x)
            boundary_ratio = float(boundary_distance / lane_width)

        distance_class = interval_distance_class(
            boundary_ratio,
            args.ratio_min,
            args.ratio_max,
        )
        interval_rows.append(
            {
                "interval": interval_index,
                "y_start": int(edges[interval_index - 1]),
                "y_end": int(edges[interval_index]),
                "center_y": int(center_y),
                "window_y_start": int(rows[0]),
                "window_y_end": int(rows[-1]),
                "boundary_x": boundary_x,
                "boundary_valid_rows": boundary_valid_rows,
                "boundary_component_count": component_count,
                "multiple_components": int(component_count > 1),
                **{f"{label}_x": lane_x[label] for label in LANE_LABELS},
                **{
                    f"{label}_valid_rows": lane_valid_rows[label]
                    for label in LANE_LABELS
                },
                "adjacent_width_count": len(adjacent_widths),
                "lane_width": lane_width,
                "rightmost_lane_x": rightmost_lane_x,
                "boundary_distance": boundary_distance,
                "boundary_ratio": boundary_ratio,
                "vertical_weight": interval_vertical_weight(
                    interval_index,
                    args.num_intervals,
                ),
                "distance_class": distance_class,
                "interval_distance_score": (
                    distance_score(
                        boundary_ratio,
                        args.ratio_min,
                        args.ratio_max,
                        args.ratio_high_decay,
                    )
                    if boundary_ratio is not None
                    else None
                ),
                "valid": int(boundary_ratio is not None),
                "invalid_reason": invalid_reason,
            }
        )

    score = score_intervals(interval_rows, args, has_prediction)
    return score, interval_rows


def prefix_from_sample_id(sample_id):
    return str(sample_id).split("_", 1)[0]


def frame_index_from_sample_id(sample_id):
    match = re.search(r"(\d+)(?:_[A-Za-z]+)?$", str(sample_id))
    return int(match.group(1)) if match else None


class CausalPresenceSmoother:
    def __init__(self, mode="none", window=3, max_frame_gap=5):
        self.mode = str(mode)
        self.window = int(window)
        self.max_frame_gap = int(max_frame_gap)
        if self.window < 1:
            raise ValueError("--presence-window must be at least 1.")
        self.values = deque(maxlen=self.window)
        self.last_prefix = None
        self.last_frame_index = None

    def update(self, sample_id, probability):
        if self.mode == "none":
            return float(probability)
        prefix = prefix_from_sample_id(sample_id)
        frame_index = frame_index_from_sample_id(sample_id)
        reset = prefix != self.last_prefix
        if (
            not reset
            and frame_index is not None
            and self.last_frame_index is not None
            and (
                frame_index <= self.last_frame_index
                or frame_index - self.last_frame_index > self.max_frame_gap
            )
        ):
            reset = True
        if reset:
            self.values.clear()
        self.values.append(float(probability))
        self.last_prefix = prefix
        self.last_frame_index = frame_index
        return float(np.median(np.asarray(self.values, dtype=np.float64)))


def draw_dashed_horizontal(image, y, color, dash=10, gap=7):
    width = image.shape[1]
    for x in range(0, width, dash + gap):
        cv2.line(image, (x, y), (min(width - 1, x + dash), y), color, 1)


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float) and not np.isfinite(value):
        return "inf"
    return f"{float(value):.{digits}f}"


def make_visualization(
    image_rgb,
    gt_mask,
    boundary_mask,
    lane_masks,
    interval_rows,
    score,
    lane_source,
    gate_accepted,
    v2_gate_accepted=None,
    presence_probability=None,
    presence_probability_raw=None,
    presence_threshold=None,
    presence_pass=None,
):
    overlay = image_rgb.copy()
    boundary_pixels = boundary_mask > 0
    boundary_color = (
        np.asarray([0, 255, 0])
        if gate_accepted
        else np.asarray([255, 0, 255])
    )
    overlay[boundary_pixels] = (
        0.45 * overlay[boundary_pixels] + 0.55 * boundary_color
    ).astype(np.uint8)

    gt_contours, _ = cv2.findContours(
        (gt_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, gt_contours, -1, (255, 0, 0), 1)

    for label, color in LANE_COLORS.items():
        ys, xs = np.where(lane_masks[label] > 0)
        overlay[ys, xs] = color

    status_color = {
        "reliable": (0, 210, 0),
        "uncertain": (255, 190, 0),
        "low_reliability": (255, 60, 60),
        "unavailable": (170, 170, 170),
        "no_prediction": (130, 130, 130),
    }[score["status"]]

    for row in interval_rows:
        y = row["center_y"]
        draw_dashed_horizontal(overlay, y, (150, 150, 150))
        for label, color in LANE_COLORS.items():
            x_value = row[f"{label}_x"]
            if x_value is not None:
                cv2.circle(overlay, (int(round(x_value)), y), 3, color, -1)
        if row["boundary_x"] is not None:
            point_color = status_color if row["valid"] else (120, 120, 120)
            cv2.circle(
                overlay,
                (int(round(row["boundary_x"])), y),
                4,
                point_color,
                -1,
            )

    panel_width = 440
    height = overlay.shape[0]
    panel = np.full((height, panel_width, 3), 24, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        panel,
        f"lane_source: {lane_source}",
        (12, 25),
        font,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    if presence_probability is not None:
        cv2.putText(
            panel,
            (
                f"V2={'PASS' if v2_gate_accepted else 'REJECT'} | "
                f"V4={float(presence_probability):.3f}/"
                f"{float(presence_threshold):.3f} "
                f"{'PASS' if presence_pass else 'REJECT'}"
            ),
            (12, 148),
            font,
            0.45,
            (90, 220, 255) if presence_pass else (255, 120, 120),
            1,
            cv2.LINE_AA,
        )
        if (
            presence_probability_raw is not None
            and abs(presence_probability_raw - presence_probability) > 1e-8
        ):
            cv2.putText(
                panel,
                f"V4 raw={float(presence_probability_raw):.3f}",
                (300, 148),
                font,
                0.40,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        panel,
        (
            f"status: {score['status']} | "
            f"gate: {'ACCEPT' if gate_accepted else 'REJECT'}"
        ),
        (12, 51),
        font,
        0.48,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"reliability: {fmt(score['reliability_score'])}",
        (12, 77),
        font,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            f"valid={score['valid_interval_count']}/{len(interval_rows)} "
            f"ratio={fmt(score['aggregate_ratio'])} CV={fmt(score['ratio_cv'])}"
        ),
        (12, 101),
        font,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            f"distance={fmt(score['distance_score'])} "
            f"pass={fmt(score['distance_pass_rate'])} "
            f"stability={fmt(score['stability_score'])}"
        ),
        (12, 124),
        font,
        0.43,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    y_text = 172 if presence_probability is not None else 149
    detail_step = min(
        25,
        max(14, (height - 205) // max(1, len(interval_rows))),
    )
    detail_scale = 0.40 if detail_step >= 20 else 0.32
    for row in sorted(interval_rows, key=lambda item: item["interval"], reverse=True):
        ratio_text = fmt(row["boundary_ratio"])
        result_text = (
            row["distance_class"] if row["valid"] else row["invalid_reason"]
        )
        detail = (
            f"I{row['interval']:02d} y={row['center_y']:3d} "
            f"ratio={ratio_text:>6} "
            f"{result_text}"
        )
        result_color = {
            "PASS": (80, 230, 80),
            "CLOSE": (255, 190, 0),
            "FAR": (255, 100, 100),
        }
        color = (
            result_color.get(row["distance_class"], (220, 220, 220))
            if row["valid"]
            else (145, 145, 145)
        )
        cv2.putText(
            panel,
            detail,
            (12, y_text),
            font,
            detail_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y_text += detail_step

    cv2.putText(
        panel,
        "raw band: green=accepted magenta=rejected | GT=red",
        (12, height - 42),
        font,
        0.35,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "lane1=yellow lane2=magenta lane3=orange lane4=cyan",
        (12, height - 18),
        font,
        0.35,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([overlay, panel], axis=1)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scene_rows(image_rows):
    groups = defaultdict(list)
    for row in image_rows:
        groups[(row["lane_source"], row["prefix"])].append(row)

    output = []
    for (lane_source, prefix), rows in sorted(groups.items()):
        statuses = Counter(row["status"] for row in rows)
        usable_scores = [
            float(row["reliability_score"])
            for row in rows
            if row["reliability_score"] not in ("", None)
        ]
        positive_scores = [
            float(row["reliability_score"])
            for row in rows
            if int(row["has_boundary"]) == 1
            and row["reliability_score"] not in ("", None)
        ]
        negative_scores = [
            float(row["reliability_score"])
            for row in rows
            if int(row["has_boundary"]) == 0
            and row["reliability_score"] not in ("", None)
        ]
        output.append(
            {
                "lane_source": lane_source,
                "prefix": prefix,
                "count": len(rows),
                "positive_count": sum(int(row["has_boundary"]) for row in rows),
                "negative_count": sum(1 - int(row["has_boundary"]) for row in rows),
                "reliable": statuses["reliable"],
                "uncertain": statuses["uncertain"],
                "low_reliability": statuses["low_reliability"],
                "unavailable": statuses["unavailable"],
                "no_prediction": statuses["no_prediction"],
                "mean_reliability_usable": (
                    float(np.mean(usable_scores)) if usable_scores else ""
                ),
                "mean_reliability_positive": (
                    float(np.mean(positive_scores)) if positive_scores else ""
                ),
                "mean_reliability_negative": (
                    float(np.mean(negative_scores)) if negative_scores else ""
                ),
            }
        )
    return output


def build_gate_metrics(image_rows, args, runtime):
    selected = [
        row
        for row in image_rows
        if row["lane_source"] == args.gate_lane_source
    ]
    positives = [row for row in selected if int(row["has_boundary"]) == 1]
    negatives = [row for row in selected if int(row["has_boundary"]) == 0]
    accepted_positives = [
        row for row in positives if int(row["gate_accepted"]) == 1
    ]
    accepted_negatives = [
        row for row in negatives if int(row["gate_accepted"]) == 1
    ]
    unavailable = [row for row in selected if row["status"] == "unavailable"]

    def ratio(numerator, denominator):
        return numerator / float(denominator) if denominator else 0.0

    def average(key, rows):
        return (
            float(np.mean([float(row[key]) for row in rows]))
            if rows
            else 0.0
        )

    def wilson_interval(successes, total, z=1.959963984540054):
        if total == 0:
            return 0.0, 0.0
        probability = successes / float(total)
        denominator = 1.0 + z * z / total
        center = (probability + z * z / (2.0 * total)) / denominator
        margin = (
            z
            * math.sqrt(
                probability * (1.0 - probability) / total
                + z * z / (4.0 * total * total)
            )
            / denominator
        )
        return center - margin, center + margin

    presence_rows = [
        row
        for row in selected
        if row.get("presence_probability") not in ("", None)
    ]
    presence_positives = [
        row for row in presence_rows if int(row["has_boundary"]) == 1
    ]
    presence_negatives = [
        row for row in presence_rows if int(row["has_boundary"]) == 0
    ]
    presence_true_positive = sum(
        int(row["presence_pass"]) for row in presence_positives
    )
    presence_true_negative = sum(
        1 - int(row["presence_pass"]) for row in presence_negatives
    )
    rejected_positive_count = len(positives) - len(accepted_positives)
    correctly_rejected_negative_count = len(negatives) - len(accepted_negatives)
    final_empty_output_count = (
        rejected_positive_count + correctly_rejected_negative_count
    )
    fpr_ci_low, fpr_ci_high = wilson_interval(
        len(accepted_negatives),
        len(negatives),
    )

    return {
        "gate_mode": args.gate_mode,
        "gate_lane_source": args.gate_lane_source,
        "presence_verifier_enabled": bool(presence_rows),
        "presence_threshold": (
            float(presence_rows[0]["presence_threshold"])
            if presence_rows
            else None
        ),
        "presence_recall": ratio(
            presence_true_positive,
            len(presence_positives),
        ),
        "presence_specificity": ratio(
            presence_true_negative,
            len(presence_negatives),
        ),
        "samples": len(selected),
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "accepted_positive_samples": len(accepted_positives),
        "accepted_negative_samples": len(accepted_negatives),
        "positive_acceptance_rate": ratio(
            len(accepted_positives),
            len(positives),
        ),
        "negative_rejection_rate": ratio(
            len(negatives) - len(accepted_negatives),
            len(negatives),
        ),
        "post_gate_false_positive_rate": ratio(
            len(accepted_negatives),
            len(negatives),
        ),
        "frame_level_fpr_wilson95_low": fpr_ci_low,
        "frame_level_fpr_wilson95_high": fpr_ci_high,
        "accepted_boundary_f1": average(
            "gate_boundary_f1",
            accepted_positives,
        ),
        "accepted_mean_boundary_error_px": average(
            "gate_mean_boundary_error",
            accepted_positives,
        ),
        "accepted_max_boundary_error_px": average(
            "gate_max_boundary_error",
            accepted_positives,
        ),
        "end_to_end_positive_boundary_f1": average(
            "gate_boundary_f1",
            positives,
        ),
        "end_to_end_positive_mean_boundary_error_px": average(
            "gate_mean_boundary_error",
            positives,
        ),
        "v2_geometry_unavailable_samples": len(unavailable),
        "v2_geometry_unavailable_ratio": ratio(
            len(unavailable), len(selected)
        ),
        "rejected_positive_samples": rejected_positive_count,
        "correctly_rejected_negative_samples": (
            correctly_rejected_negative_count
        ),
        "final_empty_output_samples": final_empty_output_count,
        "final_empty_output_ratio": ratio(
            final_empty_output_count, len(selected)
        ),
        # Backward-compatible aliases for existing report readers.
        "unavailable_samples": len(unavailable),
        "unavailable_ratio": ratio(len(unavailable), len(selected)),
        "full_pipeline_seconds": runtime.get(
            "full_pipeline_seconds",
            0.0,
        ),
        "full_pipeline_mean_ms": runtime.get(
            "full_pipeline_mean_ms",
            0.0,
        ),
        "full_pipeline_fps": runtime.get("full_pipeline_fps", 0.0),
    }


def write_gate_metrics(path, metrics):
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        path.with_suffix(".csv"),
        [metrics],
        list(metrics.keys()),
    )


def write_markdown_summary(
    path,
    image_rows,
    scene_rows,
    args,
    runtime,
    gate_metrics,
):
    gate_description = (
        "V2 validates boundary geometry and V4 validates emergency-shoulder "
        "presence. The final output requires both gates to pass; neither "
        "module moves, fits, or smooths the predicted boundary."
        if gate_metrics.get("presence_verifier_enabled")
        else (
            "V2 validates the raw boundary and gates the final system output. "
            "It does not move, fit, or smooth the predicted boundary."
        )
    )
    lines = [
        "# Lane-Boundary Reliability Gate Evaluation",
        "",
        gate_description,
        "",
        "## Configuration",
        "",
        f"- Image size: 800x450",
        f"- ROI: {args.roi_start:.2f}H to {args.roi_end:.2f}H",
        f"- Intervals: {args.num_intervals}",
        f"- Sampling mode: {args.sampling_mode}",
        f"- Perspective gamma: {args.perspective_gamma}",
        f"- Window: center +/- {args.window_radius} rows",
        f"- Minimum valid rows: {args.min_valid_rows}",
        f"- Band threshold: {args.band_threshold}",
        f"- Flat valid ratio range: [{args.ratio_min}, {args.ratio_max}]",
        f"- Stable CV: <= {args.stable_cv}",
        "- Final score: 0.5 distance + 0.3 stability + 0.2 coverage",
        (
            f"- Bottom-up support: I{args.num_intervals:02d} weight 1.0, "
            "decreasing to I01 weight 0.5"
        ),
        "- Final adjustment: base score x (0.5 + 0.5 x weighted pass rate)",
        "- Three valid intervals are scoreable with coverage 0.4",
        f"- Gate mode: {args.gate_mode}",
        f"- Formal gate lane source: {args.gate_lane_source}",
        "",
        "## Overall",
        "",
        "| Lane source | Images | Reliable | Uncertain | Low | V2 geometry unavailable | No prediction | Mean usable score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane_source in sorted({row["lane_source"] for row in image_rows}):
        selected = [row for row in image_rows if row["lane_source"] == lane_source]
        statuses = Counter(row["status"] for row in selected)
        scores = [
            float(row["reliability_score"])
            for row in selected
            if row["reliability_score"] not in ("", None)
        ]
        lines.append(
            f"| {lane_source} | {len(selected)} | {statuses['reliable']} | "
            f"{statuses['uncertain']} | {statuses['low_reliability']} | "
            f"{statuses['unavailable']} | {statuses['no_prediction']} | "
            f"{np.mean(scores):.4f} |" if scores else
            f"| {lane_source} | {len(selected)} | {statuses['reliable']} | "
            f"{statuses['uncertain']} | {statuses['low_reliability']} | "
            f"{statuses['unavailable']} | {statuses['no_prediction']} | - |"
        )

    lines.extend(
        [
            "",
            "## Formal Output Gate",
            "",
            "| Metric | Value |",
            "|---|---:|",
            (
                "| Positive acceptance rate | "
                f"{100.0 * gate_metrics['positive_acceptance_rate']:.2f}% |"
            ),
            (
                "| Negative rejection rate | "
                f"{100.0 * gate_metrics['negative_rejection_rate']:.2f}% |"
            ),
            (
                "| Post-gate false-positive rate | "
                f"{100.0 * gate_metrics['post_gate_false_positive_rate']:.2f}% |"
            ),
            (
                "| V4 presence recall | "
                f"{100.0 * gate_metrics['presence_recall']:.2f}% |"
            ),
            (
                "| V4 presence specificity | "
                f"{100.0 * gate_metrics['presence_specificity']:.2f}% |"
            ),
            (
                "| Accepted-boundary F1 | "
                f"{gate_metrics['accepted_boundary_f1']:.6f} |"
            ),
            (
                "| Accepted-boundary CMBE | "
                f"{gate_metrics['accepted_mean_boundary_error_px']:.4f} px |"
            ),
            (
                "| V2 geometry-state Unavailable | "
                f"{100.0 * gate_metrics['v2_geometry_unavailable_ratio']:.2f}% |"
            ),
            (
                "| Final positive rejection rate | "
                f"{100.0 * (1.0 - gate_metrics['positive_acceptance_rate']):.2f}% |"
            ),
            (
                "| Final correctly rejected negative rate | "
                f"{100.0 * gate_metrics['negative_rejection_rate']:.2f}% |"
            ),
            (
                "| Final all-sample empty-output ratio | "
                f"{100.0 * gate_metrics['final_empty_output_ratio']:.2f}% |"
            ),
            (
                "| Frame-level FPR Wilson 95% CI | "
                f"[{100.0 * gate_metrics['frame_level_fpr_wilson95_low']:.2f}%, "
                f"{100.0 * gate_metrics['frame_level_fpr_wilson95_high']:.2f}%] |"
            ),
            (
                "| Full-pipeline FPS | "
                f"{gate_metrics['full_pipeline_fps']:.2f} |"
            ),
            "",
            "## Runtime",
            "",
            f"- Boundary inference total: {runtime['boundary_inference_seconds']:.4f} s",
            f"- V4 presence inference total: {runtime['presence_inference_seconds']:.4f} s",
            f"- Lane-model inference total: {runtime['lane_model_inference_seconds']:.4f} s",
            f"- Reliability scoring total: {runtime['scoring_seconds']:.4f} s",
            f"- Output gating total: {runtime['gating_seconds']:.4f} s",
            f"- Scoring mean per source/sample: {runtime['scoring_mean_ms']:.4f} ms",
            f"- Full pipeline mean: {runtime['full_pipeline_mean_ms']:.4f} ms/image",
            f"- Full pipeline FPS: {runtime['full_pipeline_fps']:.2f}",
            "",
            "## Files",
            "",
            "- `image_summary.csv`: one row per image and lane source.",
            (
                f"- `interval_details.csv`: {args.num_intervals} interval rows "
                "per image and lane source."
            ),
            "- `scene_summary.csv`: per-prefix summaries.",
            "- `system_gate_metrics.json/csv`: formal system-level gate metrics.",
            "- `gated_masks_<source>/`: final right-boundary outputs after gating.",
            "- `visualizations_labelme/`: full split with LabelMe lane coordinates.",
            "- `visualizations_model/`: full split with lane-model coordinates.",
            "- `runtime_benchmark.json`: measured runtime components.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    lane_model_path = Path(args.lane_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.gate_lane_source not in args.lane_sources:
        raise ValueError(
            "--gate-lane-source must also be included in --lane-sources."
        )

    if args.boundary_backend == "native":
        with config_path.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
    else:
        cfg = {
            "dataset": {
                "root": "datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601",
                "image_size": [450, 800],
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            }
        }
    image_size = tuple(cfg["dataset"]["image_size"])
    if image_size != (450, 800):
        raise ValueError(
            f"This evaluator requires dataset.image_size [450, 800], got {list(image_size)}"
        )

    device = resolve_device(args.device)
    if args.boundary_backend == "native":
        boundary_model = load_boundary_model(
            cfg, checkpoint_path, device, deploy_band_only=args.deploy_band_only
        )
    else:
        boundary_model = load_mmseg_boundary_model(
            config_path, checkpoint_path, device
        )

    presence_verifier = None
    presence_threshold = None
    captured_backbone_features = {}
    backbone_hook = None
    if args.presence_verifier:
        if args.boundary_backend != "native":
            raise ValueError(
                "V4 presence verification currently requires --boundary-backend native."
            )
        presence_checkpoint_path = Path(args.presence_verifier)
        presence_verifier, stored_threshold, presence_checkpoint = (
            load_v4_presence_verifier(presence_checkpoint_path, device)
        )
        presence_threshold = (
            float(args.presence_threshold)
            if args.presence_threshold >= 0.0
            else float(stored_threshold)
        )
        expected_boundary_checkpoint = str(
            presence_checkpoint.get("boundary_checkpoint", "")
        ).strip()
        if expected_boundary_checkpoint:
            expected_path = Path(expected_boundary_checkpoint).resolve()
            if expected_path != checkpoint_path.resolve():
                raise ValueError(
                    "V4 verifier was trained with a different boundary checkpoint: "
                    f"{expected_path} != {checkpoint_path.resolve()}"
                )

        def capture_backbone_features(_module, _inputs, output):
            captured_backbone_features["f4"] = output[-1]

        backbone_hook = boundary_model.backbone.register_forward_hook(
            capture_backbone_features
        )
    if args.presence_smoothing != "none" and presence_verifier is None:
        raise ValueError(
            "--presence-smoothing requires --presence-verifier."
        )
    presence_smoother = CausalPresenceSmoother(
        mode=args.presence_smoothing,
        window=args.presence_window,
        max_frame_gap=args.presence_max_frame_gap,
    )

    dataset_cfg = cfg["dataset"]
    split_csv = f"splits/{args.split}.csv"
    dataset = BoundaryDataset(
        dataset_cfg["root"],
        split_csv,
        dataset_cfg["image_size"],
        dataset_cfg["image_mean"],
        dataset_cfg["image_std"],
        train=False,
    )
    indices = list(range(len(dataset)))
    if args.prefixes:
        selected_prefixes = {str(value) for value in args.prefixes}
        indices = [
            index
            for index, row in enumerate(dataset.rows)
            if prefix_from_sample_id(Path(row["image"]).stem) in selected_prefixes
        ]
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    loader_dataset = Subset(dataset, indices)
    loader = DataLoader(
        loader_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    rows_by_id = {
        Path(row["image"]).stem: row
        for row in dataset.rows
    }

    lane_model = None
    lane_cls_num = None
    lane_args = None
    if "model" in args.lane_sources:
        if not lane_model_path.exists():
            raise FileNotFoundError(f"Lane model not found: {lane_model_path}")
        lane_model, lane_cls_num, lane_args = load_project_lane_model(
            lane_model_path,
            device,
        )

    runtime = {
        "boundary_inference_seconds": 0.0,
        "presence_inference_seconds": 0.0,
        "lane_model_inference_seconds": 0.0,
        "scoring_seconds": 0.0,
        "gating_seconds": 0.0,
        "visualization_seconds": 0.0,
        "processed_images": 0,
        "scored_source_samples": 0,
    }
    scoring_seconds_by_source = defaultdict(float)
    gating_seconds_by_source = defaultdict(float)
    image_rows = []
    interval_output_rows = []
    hard_negative_targets = {}

    print(f"DEVICE={device}")
    print(f"BOUNDARY_BACKEND={args.boundary_backend}")
    print(f"DEPLOY_BAND_ONLY={args.deploy_band_only}")
    print(f"CONFIG={config_path.resolve()}")
    print(f"BOUNDARY_CHECKPOINT={checkpoint_path.resolve()}")
    if presence_verifier is not None:
        print(f"PRESENCE_VERIFIER={Path(args.presence_verifier).resolve()}")
        print(f"PRESENCE_THRESHOLD={presence_threshold:.6f}")
        print(
            f"PRESENCE_SMOOTHING={args.presence_smoothing} "
            f"WINDOW={args.presence_window}"
        )
    print(f"LANE_SOURCES={','.join(args.lane_sources)}")
    if lane_model is not None:
        print(f"LANE_MODEL={lane_model_path.resolve()}")
    print(f"SPLIT={args.split} IMAGES={len(loader_dataset)}")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"reliability {args.split}", ncols=100):
            images = batch["image"].to(device)

            synchronize(device)
            start = time.perf_counter()
            outputs = boundary_model(images)
            probabilities = torch.sigmoid(band_logits(outputs)).detach().cpu().numpy()
            synchronize(device)
            runtime["boundary_inference_seconds"] += time.perf_counter() - start

            if presence_verifier is not None:
                if "f4" not in captured_backbone_features:
                    raise RuntimeError(
                        "V4 could not capture the boundary backbone F4 feature."
                    )
                synchronize(device)
                start = time.perf_counter()
                presence_probabilities = torch.sigmoid(
                    presence_verifier(captured_backbone_features["f4"])
                ).detach().cpu().numpy()
                synchronize(device)
                runtime["presence_inference_seconds"] += (
                    time.perf_counter() - start
                )
            else:
                presence_probabilities = np.full(
                    images.shape[0],
                    np.nan,
                    dtype=np.float32,
                )

            gt_masks = batch["mask"].cpu().numpy()
            for batch_index, sample_id in enumerate(batch["id"]):
                sample_id = str(sample_id)
                dataset_row = rows_by_id[sample_id]
                image_rgb = denormalize_image(batch["image"][batch_index])
                height, width = image_rgb.shape[:2]
                gt_mask = gt_masks[batch_index, 0] > 0.5
                boundary_probability = probabilities[batch_index, 0]
                boundary_mask = boundary_probability >= args.band_threshold
                if args.save_raw_boundary_masks:
                    raw_boundary_dir = out_dir / "raw_boundary_masks"
                    raw_boundary_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(
                        boundary_mask.astype(np.uint8) * 255,
                    ).save(raw_boundary_dir / f"{sample_id}.png")
                presence_probability_raw = (
                    float(presence_probabilities[batch_index])
                    if presence_verifier is not None
                    else None
                )
                presence_probability = (
                    presence_smoother.update(
                        sample_id,
                        presence_probability_raw,
                    )
                    if presence_probability_raw is not None
                    else None
                )
                presence_pass = (
                    presence_probability >= presence_threshold
                    if presence_verifier is not None
                    else True
                )

                lane_shapes_by_source = {}
                if "labelme" in args.lane_sources:
                    lane_shapes_by_source["labelme"] = load_labelme_lanes(
                        dataset_row["json"]
                    )
                if "model" in args.lane_sources:
                    synchronize(device)
                    start = time.perf_counter()
                    lane_shapes_by_source["model"] = predicted_lanes_as_shapes(
                        lane_model,
                        image_rgb,
                        lane_args,
                        lane_cls_num,
                        device,
                    )
                    synchronize(device)
                    runtime["lane_model_inference_seconds"] += (
                        time.perf_counter() - start
                    )

                for lane_source, lane_shapes in lane_shapes_by_source.items():
                    lane_masks = rasterize_lanes(lane_shapes, height, width)
                    start = time.perf_counter()
                    score, interval_rows = evaluate_sample(
                        boundary_mask,
                        lane_masks,
                        args,
                    )
                    scoring_elapsed = time.perf_counter() - start
                    runtime["scoring_seconds"] += scoring_elapsed
                    scoring_seconds_by_source[lane_source] += scoring_elapsed
                    runtime["scored_source_samples"] += 1

                    start = time.perf_counter()
                    v2_gate_accepted = gate_accepts(
                        score["status"],
                        args.gate_mode,
                    )
                    gate_accepted = bool(
                        v2_gate_accepted and presence_pass
                    )
                    gated_mask = (
                        boundary_mask.copy()
                        if gate_accepted
                        else np.zeros_like(boundary_mask, dtype=bool)
                    )
                    gating_elapsed = time.perf_counter() - start
                    runtime["gating_seconds"] += gating_elapsed
                    gating_seconds_by_source[lane_source] += gating_elapsed

                    gate_boundary = boundary_metrics(
                        gated_mask.astype(np.uint8),
                        gt_mask.astype(np.uint8),
                        args.boundary_tolerance,
                    )

                    summary_row = {
                        "sample_id": sample_id,
                        "prefix": prefix_from_sample_id(sample_id),
                        "lane_source": lane_source,
                        "has_boundary": int(dataset_row["has_boundary"]),
                        "band_prediction_pixels": int(boundary_mask.sum()),
                        "v2_gate_accepted": int(v2_gate_accepted),
                        "presence_probability": (
                            presence_probability
                            if presence_probability is not None
                            else ""
                        ),
                        "presence_probability_raw": (
                            presence_probability_raw
                            if presence_probability_raw is not None
                            else ""
                        ),
                        "presence_threshold": (
                            presence_threshold
                            if presence_threshold is not None
                            else ""
                        ),
                        "presence_pass": (
                            int(presence_pass)
                            if presence_verifier is not None
                            else ""
                        ),
                        "gate_accepted": int(gate_accepted),
                        "gate_output_pixels": int(gated_mask.sum()),
                        "gate_boundary_f1": gate_boundary["boundary_f1"],
                        "gate_mean_boundary_error": gate_boundary[
                            "mean_boundary_error"
                        ],
                        "gate_max_boundary_error": gate_boundary[
                            "max_boundary_error"
                        ],
                        **score,
                    }
                    image_rows.append(summary_row)
                    if not args.skip_gated_masks:
                        gated_dir = out_dir / f"gated_masks_{lane_source}"
                        gated_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(
                            gated_mask.astype(np.uint8) * 255,
                        ).save(gated_dir / f"{sample_id}.png")
                    if (
                        args.export_hard_negative_csv
                        and lane_source == "model"
                        and int(dataset_row["has_boundary"]) == 0
                        and boundary_mask.any()
                    ):
                        reliability_value = score["reliability_score"]
                        reliability_score = (
                            float(reliability_value)
                            if reliability_value is not None
                            else None
                        )
                        mining_weight = 0.0
                        if (
                            score["status"] in {"reliable", "uncertain"}
                            and reliability_score is not None
                            and reliability_score
                            >= float(args.hard_negative_min_score)
                        ):
                            mining_weight = float(
                                np.clip(
                                    (reliability_score - 0.5) / 0.5,
                                    0.0,
                                    1.0,
                                )
                            )
                        elif score["status"] == "low_reliability":
                            mining_weight = float(
                                args.hard_negative_low_weight
                            )
                        elif score["status"] == "unavailable":
                            mining_weight = float(
                                args.hard_negative_unavailable_weight
                            )
                        if mining_weight > 0:
                            hard_negative_targets[sample_id] = {
                                "mask": boundary_mask.astype(np.uint8) * 255,
                                "weight": mining_weight,
                                "score": reliability_score,
                                "status": score["status"],
                            }
                    for interval_row in interval_rows:
                        interval_output_rows.append(
                            {
                                "sample_id": sample_id,
                                "prefix": prefix_from_sample_id(sample_id),
                                "lane_source": lane_source,
                                "has_boundary": int(dataset_row["has_boundary"]),
                                **interval_row,
                            }
                        )

                    if not args.skip_visualizations:
                        start = time.perf_counter()
                        visualization = make_visualization(
                            image_rgb,
                            gt_mask,
                            boundary_mask,
                            lane_masks,
                            interval_rows,
                            score,
                            lane_source,
                            gate_accepted,
                            v2_gate_accepted=v2_gate_accepted,
                            presence_probability=presence_probability,
                            presence_probability_raw=presence_probability_raw,
                            presence_threshold=presence_threshold,
                            presence_pass=presence_pass,
                        )
                        visualization_dir = out_dir / f"visualizations_{lane_source}"
                        visualization_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(visualization).save(
                            visualization_dir / f"{sample_id}_reliability.jpg",
                            quality=92,
                        )
                        runtime["visualization_seconds"] += time.perf_counter() - start
                runtime["processed_images"] += 1

    runtime["scoring_mean_ms"] = (
        1000.0 * runtime["scoring_seconds"] / runtime["scored_source_samples"]
        if runtime["scored_source_samples"]
        else 0.0
    )
    runtime["boundary_inference_mean_ms"] = (
        1000.0 * runtime["boundary_inference_seconds"] / runtime["processed_images"]
        if runtime["processed_images"]
        else 0.0
    )
    runtime["lane_model_inference_mean_ms"] = (
        1000.0 * runtime["lane_model_inference_seconds"] / runtime["processed_images"]
        if runtime["processed_images"] and "model" in args.lane_sources
        else 0.0
    )
    gate_source_scoring = scoring_seconds_by_source[args.gate_lane_source]
    gate_source_gating = gating_seconds_by_source[args.gate_lane_source]
    gate_source_lane = (
        runtime["lane_model_inference_seconds"]
        if args.gate_lane_source == "model"
        else 0.0
    )
    runtime["full_pipeline_seconds"] = (
        runtime["boundary_inference_seconds"]
        + runtime["presence_inference_seconds"]
        + gate_source_lane
        + gate_source_scoring
        + gate_source_gating
    )
    runtime["full_pipeline_mean_ms"] = (
        1000.0
        * runtime["full_pipeline_seconds"]
        / runtime["processed_images"]
        if runtime["processed_images"]
        else 0.0
    )
    runtime["full_pipeline_fps"] = (
        runtime["processed_images"] / runtime["full_pipeline_seconds"]
        if runtime["full_pipeline_seconds"] > 0
        else 0.0
    )

    image_fields = [
        "sample_id",
        "prefix",
        "lane_source",
        "has_boundary",
        "band_prediction_pixels",
        "v2_gate_accepted",
        "presence_probability",
        "presence_probability_raw",
        "presence_threshold",
        "presence_pass",
        "gate_accepted",
        "gate_output_pixels",
        "gate_boundary_f1",
        "gate_mean_boundary_error",
        "gate_max_boundary_error",
        "valid_interval_count",
        "aggregate_ratio",
        "distance_score",
        "ratio_cv",
        "stability_score",
        "coverage_score",
        "distance_pass_rate",
        "base_reliability_score",
        "reliability_score",
        "status",
    ]
    interval_fields = [
        "sample_id",
        "prefix",
        "lane_source",
        "has_boundary",
        "interval",
        "y_start",
        "y_end",
        "center_y",
        "window_y_start",
        "window_y_end",
        "boundary_x",
        "boundary_valid_rows",
        "boundary_component_count",
        "multiple_components",
        *[f"{label}_x" for label in LANE_LABELS],
        *[f"{label}_valid_rows" for label in LANE_LABELS],
        "adjacent_width_count",
        "lane_width",
        "rightmost_lane_x",
        "boundary_distance",
        "boundary_ratio",
        "vertical_weight",
        "distance_class",
        "interval_distance_score",
        "valid",
        "invalid_reason",
    ]
    scene_fields = [
        "lane_source",
        "prefix",
        "count",
        "positive_count",
        "negative_count",
        "reliable",
        "uncertain",
        "low_reliability",
        "unavailable",
        "no_prediction",
        "mean_reliability_usable",
        "mean_reliability_positive",
        "mean_reliability_negative",
    ]

    scene_rows = build_scene_rows(image_rows)
    gate_metrics = build_gate_metrics(image_rows, args, runtime)
    write_csv(out_dir / "image_summary.csv", image_rows, image_fields)
    write_csv(
        out_dir / "interval_details.csv",
        interval_output_rows,
        interval_fields,
    )
    write_csv(out_dir / "scene_summary.csv", scene_rows, scene_fields)
    (out_dir / "runtime_benchmark.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_gate_metrics(
        out_dir / "system_gate_metrics.json",
        gate_metrics,
    )
    write_markdown_summary(
        out_dir / "evaluation_summary.md",
        image_rows,
        scene_rows,
        args,
        runtime,
        gate_metrics,
    )
    if args.export_hard_negative_csv:
        if "model" not in args.lane_sources:
            raise ValueError(
                "--export-hard-negative-csv requires --lane-sources model."
            )
        export_csv = Path(args.export_hard_negative_csv)
        mask_dir = (
            Path(args.hard_negative_mask_dir)
            if args.hard_negative_mask_dir
            else export_csv.parent / "hard_negative_masks_v2"
        )
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        exported_rows = []
        fieldnames = list(dataset.rows[0].keys())
        for field in (
            "hard_negative_mask",
            "hard_negative_weight",
            "hard_negative_v2_score",
            "hard_negative_v2_status",
        ):
            if field not in fieldnames:
                fieldnames.append(field)
        for source_row in dataset.rows:
            row = dict(source_row)
            sample_id = Path(row["image"]).stem
            target = hard_negative_targets.get(sample_id)
            row["hard_negative_mask"] = ""
            row["hard_negative_weight"] = "0"
            row["hard_negative_v2_score"] = ""
            row["hard_negative_v2_status"] = ""
            if target is not None:
                mask_path = mask_dir / f"{sample_id}.png"
                Image.fromarray(target["mask"], mode="L").save(mask_path)
                row["hard_negative_mask"] = str(mask_path.resolve())
                row["hard_negative_weight"] = f"{target['weight']:.8f}"
                row["hard_negative_v2_score"] = (
                    f"{target['score']:.8f}"
                    if target["score"] is not None
                    else ""
                )
                row["hard_negative_v2_status"] = str(target["status"])
            exported_rows.append(row)
        write_csv(export_csv, exported_rows, fieldnames)
        print(f"HARD_NEGATIVE_CSV={export_csv.resolve()}")
        print(f"HARD_NEGATIVE_TARGETS={len(hard_negative_targets)}")

    print("DONE")
    print(f"OUTPUT={out_dir.resolve()}")
    print(f"IMAGES={runtime['processed_images']}")
    print(f"SCORING_MEAN_MS={runtime['scoring_mean_ms']:.4f}")
    print(f"BOUNDARY_INFERENCE_MEAN_MS={runtime['boundary_inference_mean_ms']:.4f}")
    if "model" in args.lane_sources:
        print(f"LANE_MODEL_INFERENCE_MEAN_MS={runtime['lane_model_inference_mean_ms']:.4f}")
    print(
        f"POSITIVE_ACCEPTANCE_RATE="
        f"{gate_metrics['positive_acceptance_rate']:.6f}"
    )
    print(
        f"NEGATIVE_REJECTION_RATE="
        f"{gate_metrics['negative_rejection_rate']:.6f}"
    )
    print(
        f"POST_GATE_FALSE_POSITIVE_RATE="
        f"{gate_metrics['post_gate_false_positive_rate']:.6f}"
    )
    if backbone_hook is not None:
        backbone_hook.remove()
    print(f"FULL_PIPELINE_FPS={runtime['full_pipeline_fps']:.4f}")


if __name__ == "__main__":
    main()
