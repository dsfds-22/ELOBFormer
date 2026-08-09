"""Run the frozen V4.3 right-boundary post-processing pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "system" / "v4_3_final_gate.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen V4.3 pipeline. Model thresholds and gate parameters "
            "come only from the frozen config and cannot be overridden here."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prefixes", nargs="+", default=[])
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--skip-gated-masks", action="store_true")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(value, label):
    path = project_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def main():
    args = parse_args()
    config_path = require_file(args.config, "Frozen V4.3 config")
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    if cfg.get("version") != "V4.3":
        raise ValueError(f"Expected V4.3 config, got: {cfg.get('version')!r}")
    if cfg.get("status") != "frozen_final_postprocessor":
        raise ValueError("The selected config is not marked as frozen.")

    boundary = cfg["boundary_model"]
    lane = cfg["lane_model"]
    geometry = cfg["v2_geometry_gate"]
    presence = cfg["v4_3_presence_gate"]
    evaluation = cfg["evaluation"]

    boundary_config = require_file(boundary["config"], "Boundary model config")
    boundary_checkpoint = require_file(
        boundary["checkpoint"], "Boundary checkpoint"
    )
    lane_checkpoint = require_file(lane["checkpoint"], "Lane checkpoint")
    presence_checkpoint = require_file(
        presence["checkpoint"], "V4 presence checkpoint"
    )
    require_file(
        presence["threshold_calibration"], "Validation calibration record"
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "evaluate_lane_boundary_reliability.py"),
        "--config",
        str(boundary_config),
        "--boundary-backend",
        str(boundary["backend"]),
        "--checkpoint",
        str(boundary_checkpoint),
        "--split",
        args.split,
        "--out-dir",
        str(project_path(args.out_dir)),
        "--lane-sources",
        str(lane["source"]),
        "--lane-model",
        str(lane_checkpoint),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--band-threshold",
        str(boundary["band_threshold"]),
        "--roi-start",
        str(geometry["roi_start"]),
        "--roi-end",
        str(geometry["roi_end"]),
        "--num-intervals",
        str(geometry["num_intervals"]),
        "--sampling-mode",
        str(geometry["sampling_mode"]),
        "--perspective-gamma",
        str(geometry["perspective_gamma"]),
        "--window-radius",
        str(geometry["window_radius"]),
        "--min-valid-rows",
        str(geometry["min_valid_rows"]),
        "--ratio-min",
        str(geometry["ratio_min"]),
        "--ratio-max",
        str(geometry["ratio_max"]),
        "--ratio-high-decay",
        str(geometry["ratio_high_decay"]),
        "--stable-cv",
        str(geometry["stable_cv"]),
        "--cv-decay",
        str(geometry["cv_decay"]),
        "--gate-mode",
        str(geometry["gate_mode"]),
        "--gate-lane-source",
        str(lane["source"]),
        "--boundary-tolerance",
        str(evaluation["boundary_tolerance"]),
        "--presence-verifier",
        str(presence_checkpoint),
        "--presence-threshold",
        str(presence["threshold"]),
        "--presence-smoothing",
        str(presence["smoothing"]),
        "--presence-window",
        str(presence["window"]),
        "--presence-max-frame-gap",
        str(presence["max_frame_gap"]),
        "--save-raw-boundary-masks",
    ]

    if bool(boundary.get("deploy_band_only", False)):
        command.append("--deploy-band-only")

    if args.max_samples > 0:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.prefixes:
        command.extend(["--prefixes", *args.prefixes])
    if args.skip_visualizations:
        command.append("--skip-visualizations")
    if args.skip_gated_masks:
        command.append("--skip-gated-masks")

    print("Frozen V4.3 command:")
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
