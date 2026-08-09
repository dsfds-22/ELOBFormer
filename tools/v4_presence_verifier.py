"""Lightweight semantic presence verifier for the V4 output gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F


POOL_REGIONS = ("global", "right_half", "lower_right")


def pool_presence_features(feature_map):
    """Pool global and right-side semantic evidence from a frozen F4 map."""
    if feature_map.ndim != 4:
        raise ValueError(
            f"Expected BCHW feature map, got shape {tuple(feature_map.shape)}"
        )
    _, _, height, width = feature_map.shape
    right_start = width // 2
    lower_start = height // 2
    regions = (
        feature_map,
        feature_map[:, :, :, right_start:],
        feature_map[:, :, lower_start:, right_start:],
    )
    return torch.cat(
        [F.adaptive_avg_pool2d(region, 1).flatten(1) for region in regions],
        dim=1,
    )


class V4PresenceVerifier(nn.Module):
    """Classify whether a task-relevant emergency shoulder exists."""

    def __init__(
        self,
        feature_channels=512,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()
        input_dim = int(feature_channels) * len(POOL_REGIONS)
        self.feature_channels = int(feature_channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, feature_map):
        features = pool_presence_features(feature_map)
        return self.classifier(features).squeeze(1)


def load_v4_presence_verifier(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_cfg = checkpoint.get("model_config", {})
    model = V4PresenceVerifier(
        feature_channels=int(model_cfg.get("feature_channels", 512)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.30)),
    ).to(device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, float(checkpoint.get("presence_threshold", 0.5)), checkpoint
