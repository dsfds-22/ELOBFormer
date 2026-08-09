"""Training hooks for external baseline initialization."""

from __future__ import annotations

from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LoadMatchingCheckpointHook(Hook):
    priority = "VERY_HIGH"

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = str(checkpoint)

    def before_train(self, runner) -> None:
        if bool(getattr(runner, "_resume", False)) or bool(
            runner.cfg.get("resume", False)
        ):
            runner.logger.info(
                "Resume is enabled; skipping transfer checkpoint loading."
            )
            return

        path = Path(self.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        model_state = model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            key = key.removeprefix("module.")
            if (
                torch.is_tensor(value)
                and key in model_state
                and value.shape == model_state[key].shape
            ):
                compatible[key] = value
            else:
                skipped.append(key)
        result = model.load_state_dict(compatible, strict=False)
        loaded_elements = sum(value.numel() for value in compatible.values())
        total_elements = sum(value.numel() for value in model_state.values())
        runner.logger.info(
            "Loaded matching transfer checkpoint %s: tensors=%d, "
            "parameter_ratio=%.6f, skipped=%d, missing=%d",
            path,
            len(compatible),
            loaded_elements / max(total_elements, 1),
            len(skipped),
            len(result.missing_keys),
        )
