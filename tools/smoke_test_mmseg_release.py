from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np

from mmcv_lite_compat import install_mmcv_lite_ext_stub
from run_mmseg_boundary import (
    install_mmengine_windows_env_guard,
    install_trusted_checkpoint_compat,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    (
        "deeplabv3plus_r50",
        "third_party/mmseg_emergency_boundary/configs/deeplabv3plus_r50_boundary_80epoch.py",
        "weights/deeplabv3plus_r50.pth",
    ),
    (
        "pidnet_s",
        "third_party/mmseg_emergency_boundary/configs/pidnet_s_boundary_80epoch.py",
        "weights/pidnet_s.pth",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=PROJECT_ROOT / "weights",
    )
    args = parser.parse_args()
    weights_dir = args.weights_dir.resolve()
    os.environ["LANE_DETECTION_PROJECT_ROOT"] = str(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    install_mmcv_lite_ext_stub()
    install_mmengine_windows_env_guard()
    install_trusted_checkpoint_compat()

    from mmengine import Config
    from mmseg.apis import inference_model, init_model

    image = np.zeros((450, 800, 3), dtype=np.uint8)
    for name, config, checkpoint in MODELS:
        cfg = Config.fromfile(str(PROJECT_ROOT / config))
        cfg.test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="PackSegInputs",
                meta_keys=(
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "scale_factor",
                ),
            ),
        ]
        model = init_model(
            cfg,
            str(weights_dir / Path(checkpoint).name),
            device="cpu",
        )
        result = inference_model(model, image)
        shape = tuple(result.pred_sem_seg.data.shape)
        print(f"OK {name}: {shape}")


if __name__ == "__main__":
    main()
