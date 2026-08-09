from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "ufld"))

from tools.auto_label_lanes_to_labelme import build_model as build_lane_model
from tools.v4_presence_verifier import load_v4_presence_verifier
from train_emergency_boundary_baseline import build_model_from_config


MODELS = (
    ("a0", "configs/main/a0_band_only.yaml", "weights/a0_band_only.pt"),
    ("a1", "configs/main/a1_band_edge03.yaml", "weights/a1_band_edge03.pt"),
    (
        "a2",
        "configs/main/a2_band_edge03_heatmap01.yaml",
        "weights/a2_band_edge03_heatmap01.pt",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=PROJECT_ROOT / "weights",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    weights_dir = args.weights_dir.resolve()

    for name, config_value, checkpoint_value in MODELS:
        cfg = yaml.safe_load((PROJECT_ROOT / config_value).read_text(encoding="utf-8"))
        model = build_model_from_config(cfg).to(device)
        checkpoint = torch.load(
            weights_dir / Path(checkpoint_value).name,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        if args.forward:
            with torch.inference_mode():
                outputs = model(torch.zeros(1, 3, 450, 800, device=device))
            keys = sorted(outputs) if isinstance(outputs, dict) else ["band"]
        else:
            keys = ["load-only"]
        print(f"OK {name}: {', '.join(keys)}")
        del model, checkpoint

    lane_args = SimpleNamespace(
        dataset="CULane",
        backbone="18",
        griding_num=200,
        num_lanes=4,
        model=str(weights_dir / "project_lane_detector.pth"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        lane_model, _ = build_lane_model(lane_args, device)
    print("OK project_lane_detector")
    del lane_model

    presence_model, threshold, _ = load_v4_presence_verifier(
        weights_dir / "v4_presence_verifier.pt",
        device,
    )
    if args.forward:
        with torch.inference_mode():
            output = presence_model(torch.zeros(1, 512, 15, 25, device=device))
        assert output.shape == (1,)
    print(f"OK v4_presence_verifier: threshold={threshold}")


if __name__ == "__main__":
    main()
