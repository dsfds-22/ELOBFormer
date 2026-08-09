"""Metrics aligned with the frozen SegFormer boundary protocol."""

from __future__ import annotations

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class EmergencyBoundaryMetric(BaseMetric):
    default_prefix = "boundary"

    def process(self, data_batch: dict, data_samples: list[dict]) -> None:
        for sample in data_samples:
            prediction = sample["pred_sem_seg"]["data"].squeeze().cpu()
            target = sample["gt_sem_seg"]["data"].squeeze().cpu()
            valid = target != 255
            prediction = (prediction > 0)[valid]
            target = (target > 0)[valid]

            intersection = int((prediction & target).sum())
            union = int((prediction | target).sum())
            prediction_sum = int(prediction.sum())
            target_sum = int(target.sum())
            self.results.append(
                (
                    intersection,
                    union,
                    prediction_sum,
                    target_sum,
                )
            )

    def compute_metrics(self, results: list[tuple]) -> dict[str, float]:
        all_iou = []
        all_dice = []
        positive_iou = []
        positive_dice = []
        positive_zero = []
        negative_false_positive = []

        for intersection, union, prediction_sum, target_sum in results:
            iou = intersection / union if union else 1.0
            denominator = prediction_sum + target_sum
            dice = 2.0 * intersection / denominator if denominator else 1.0
            all_iou.append(iou)
            all_dice.append(dice)
            if target_sum > 0:
                positive_iou.append(iou)
                positive_dice.append(dice)
                positive_zero.append(float(prediction_sum == 0))
            else:
                negative_false_positive.append(float(prediction_sum > 0))

        def mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        return {
            "iou": mean(all_iou),
            "dice": mean(all_dice),
            "positive_iou": mean(positive_iou),
            "positive_dice": mean(positive_dice),
            "positive_zero_prediction_rate": mean(positive_zero),
            "negative_false_positive_rate": mean(negative_false_positive),
            "samples": float(len(results)),
            "positive_samples": float(len(positive_iou)),
            "negative_samples": float(len(negative_false_positive)),
        }
