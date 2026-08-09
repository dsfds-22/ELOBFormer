#!/usr/bin/env python
"""Launch MMSeg train/test tools with the project's mmcv-lite support."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import yaml

from mmcv_lite_compat import install_mmcv_lite_ext_stub


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def install_mmengine_windows_env_guard() -> None:
    """Keep MMEngine environment logging from failing on Windows locales."""

    if os.name != "nt":
        return

    import torch
    import mmengine.runner.runner as runner_module

    original_collect_env = runner_module.collect_env

    def collect_env_safely():
        try:
            return original_collect_env()
        except UnicodeDecodeError as error:
            return {
                "sys.platform": sys.platform,
                "Python": sys.version.replace("\n", ""),
                "Platform": platform.platform(),
                "PyTorch": torch.__version__,
                "CUDA available": torch.cuda.is_available(),
                "Environment note": (
                    "MSVC output could not be decoded by MMEngine: "
                    f"{error}"
                ),
            }

    runner_module.collect_env = collect_env_safely


def install_trusted_checkpoint_compat() -> None:
    """Restore MMEngine checkpoint loading under PyTorch 2.6 and newer."""

    import torch

    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the released MMSegmentation boundary baselines."
    )
    parser.add_argument("task", choices=["train", "test"])
    parser.add_argument("config")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", default=[])
    args = parser.parse_args()

    os.environ["LANE_DETECTION_PROJECT_ROOT"] = str(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    install_mmcv_lite_ext_stub()
    install_mmengine_windows_env_guard()
    install_trusted_checkpoint_compat()

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(str(Path(args.config).resolve()))
    cfg.launcher = "none"
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume:
        cfg.resume = True
    if args.checkpoint:
        cfg.load_from = args.checkpoint
    if args.cfg_options:
        overrides = {}
        for item in args.cfg_options:
            if "=" not in item:
                raise ValueError(f"Invalid --cfg-options item: {item!r}")
            key, value = item.split("=", 1)
            overrides[key] = yaml.safe_load(value)
        cfg.merge_from_dict(overrides)

    runner = Runner.from_cfg(cfg)
    if args.task == "train":
        runner.train()
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for test")
        runner.test()


if __name__ == "__main__":
    main()
