"""Losses shared by external semantic-segmentation baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class BinaryBceDiceLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        loss_name: str = "loss_bce_dice",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self._loss_name = str(loss_name)
        self.eps = float(eps)

    @property
    def loss_name(self) -> str:
        return self._loss_name

    @staticmethod
    def _binary_logits(prediction: torch.Tensor) -> torch.Tensor:
        if prediction.shape[1] == 1:
            return prediction[:, 0]
        if prediction.shape[1] == 2:
            return prediction[:, 1] - prediction[:, 0]
        raise ValueError(
            "BinaryBceDiceLoss expects one or two output channels, got "
            f"{prediction.shape[1]}"
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
        **kwargs,
    ) -> torch.Tensor:
        logits = self._binary_logits(prediction)
        valid = target != int(ignore_index)
        target_binary = torch.where(
            valid, (target > 0).to(logits.dtype), torch.zeros_like(logits)
        )
        pixel_weight = valid.to(logits.dtype)
        if weight is not None:
            pixel_weight = pixel_weight * weight.to(logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits, target_binary, reduction="none"
        )
        bce = (bce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)

        probabilities = torch.sigmoid(logits) * pixel_weight
        target_binary = target_binary * pixel_weight
        dimensions = (1, 2)
        intersection = (probabilities * target_binary).sum(dimensions)
        denominator = (
            probabilities.sum(dimensions) + target_binary.sum(dimensions)
        )
        dice_loss = 1.0 - (
            (2.0 * intersection + self.eps) / (denominator + self.eps)
        ).mean()
        return self.loss_weight * (bce + dice_loss)
