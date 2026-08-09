import argparse
import csv
import io
import math
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


class BoundaryDataset(Dataset):
    def __init__(self, root, csv_path, image_size, mean, std, train=False, augmentation="basic"):
        self.root = Path(root)
        self.rows = self._read_csv(self.root / csv_path)
        self.target_h, self.target_w = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.train = train
        self.augmentation = str(augmentation or "basic").lower()

    @staticmethod
    def _read_csv(path):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def _resolve(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root / path

    def _augment_basic(self, image):
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.3:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        return image

    @staticmethod
    def _augment_gamma(image, gamma):
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
        return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    @staticmethod
    def _augment_jpeg(image, quality):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    @staticmethod
    def _augment_shadow(image):
        arr = np.asarray(image, dtype=np.float32)
        h, w = arr.shape[:2]
        x1 = random.randint(0, max(w - 1, 1))
        x2 = random.randint(0, max(w - 1, 1))
        half_width = random.randint(max(20, w // 16), max(40, w // 5))
        darkness = random.uniform(0.65, 0.88)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.line(mask, (x1, 0), (x2, h - 1), 1.0, thickness=half_width)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(8, half_width // 3))
        factor = 1.0 - mask[..., None] * (1.0 - darkness)
        arr = (arr * factor).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _augment_robust(self, image):
        image = self._augment_basic(image)
        if random.random() < 0.35:
            image = self._augment_gamma(image, random.uniform(0.8, 1.2))
        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        if random.random() < 0.20:
            image = self._augment_jpeg(image, random.randint(55, 90))
        if random.random() < 0.25:
            image = self._augment_shadow(image)
        return image

    def _augment_image(self, image):
        if self.augmentation in {"none", "off", "false", "0"}:
            return image
        if self.augmentation in {"basic", "a", "aug_a"}:
            return self._augment_basic(image)
        if self.augmentation in {"robust", "b", "aug_b"}:
            return self._augment_robust(image)
        raise ValueError(f"Unsupported augmentation preset: {self.augmentation}")

    def _pad_array(self, array, fill_value=0):
        h, w = array.shape[:2]
        if h > self.target_h or w > self.target_w:
            raise ValueError(f"Input {w}x{h} is larger than target {self.target_w}x{self.target_h}")
        if array.ndim == 2:
            out = np.full((self.target_h, self.target_w), fill_value, dtype=array.dtype)
            out[:h, :w] = array
        else:
            out = np.full((self.target_h, self.target_w, array.shape[2]), fill_value, dtype=array.dtype)
            out[:h, :w, :] = array
        valid = np.zeros((self.target_h, self.target_w), dtype=np.float32)
        valid[:h, :w] = 1.0
        return out, valid

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self._resolve(row["image"])
        band_path = self._resolve(row["band_mask"])
        edge_path = self._resolve(row.get("edge_mask", row["band_mask"]))
        heatmap_value = str(row.get("heatmap", "")).strip()

        image = Image.open(image_path).convert("RGB")
        if self.train:
            image = self._augment_image(image)
        image_np = np.asarray(image, dtype=np.uint8)
        band_np = np.asarray(Image.open(band_path).convert("L"), dtype=np.uint8)
        edge_np = np.asarray(Image.open(edge_path).convert("L"), dtype=np.uint8)
        if heatmap_value:
            heatmap_path = self._resolve(heatmap_value)
            heatmap_np = np.asarray(
                Image.open(heatmap_path).convert("L"),
                dtype=np.uint8,
            )
        else:
            heatmap_np = np.zeros_like(band_np, dtype=np.uint8)
        image_np, valid_np = self._pad_array(image_np, fill_value=0)
        band_np, _ = self._pad_array(band_np, fill_value=0)
        edge_np, _ = self._pad_array(edge_np, fill_value=0)
        heatmap_np, _ = self._pad_array(heatmap_np, fill_value=0)

        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - self.mean) / self.std
        band_t = torch.from_numpy(band_np).unsqueeze(0).float() / 255.0
        edge_t = torch.from_numpy(edge_np).unsqueeze(0).float() / 255.0
        heatmap_t = (
            torch.from_numpy(heatmap_np).unsqueeze(0).float() / 255.0
        )
        valid_t = torch.from_numpy(valid_np).unsqueeze(0).float()

        sample = {
            "image": image_t,
            "mask": band_t,
            "band": band_t,
            "edge": edge_t,
            "heatmap": heatmap_t,
            "valid": valid_t,
            "id": Path(row["image"]).stem,
            "has_boundary": torch.tensor(int(row["has_boundary"]), dtype=torch.long),
        }
        return sample


def build_train_sampler(dataset, sampling_cfg, seed):
    mode = str(sampling_cfg.get("mode", "random")).lower()
    if mode in {"random", "shuffle", "none", "off"}:
        return None, None
    if mode != "sqrt_scene":
        raise ValueError(f"Unsupported sampling mode: {mode}")

    group_fields = sampling_cfg.get(
        "group_fields",
        ["reviewed_segment_id", "capture_session", "prefix"],
    )
    group_keys = []
    for row in dataset.rows:
        group_key = ""
        for field in group_fields:
            value = str(row.get(field, "")).strip()
            if value:
                group_key = f"{field}:{value}"
                break
        if not group_key:
            group_key = f"sample:{row.get('sample_id', len(group_keys))}"
        group_keys.append(group_key)

    group_counts = Counter(group_keys)
    weights = torch.tensor(
        [1.0 / math.sqrt(group_counts[key]) for key in group_keys],
        dtype=torch.double,
    )
    num_samples = int(sampling_cfg.get("num_samples", 0)) or len(dataset)
    replacement = bool(sampling_cfg.get("replacement", True))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )
    stats = {
        "mode": mode,
        "groups": len(group_counts),
        "num_samples": num_samples,
        "replacement": replacement,
        "smallest_group": min(group_counts.values()),
        "largest_group": max(group_counts.values()),
        "group_fields": list(group_fields),
    }
    return sampler, stats


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x_tokens = x.flatten(2).transpose(1, 2)
        x_tokens = self.norm(x_tokens)
        x = x_tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio=sr_ratio, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class MixVisionTransformerB1(nn.Module):
    def __init__(self, drop=0.0, variant="b1"):
        super().__init__()
        dims = [64, 128, 320, 512]
        variant = str(variant).lower()
        if variant == "b2":
            depths = [3, 4, 6, 3]
        elif variant == "b1":
            depths = [2, 2, 2, 2]
        else:
            raise ValueError(f"Unsupported SegFormer variant: {variant}")
        heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        self.embed_dims = dims
        self.variant = variant

        self.patch_embed1 = OverlapPatchEmbed(3, dims[0], 7, 4, 3)
        self.patch_embed2 = OverlapPatchEmbed(dims[0], dims[1], 3, 2, 1)
        self.patch_embed3 = OverlapPatchEmbed(dims[1], dims[2], 3, 2, 1)
        self.patch_embed4 = OverlapPatchEmbed(dims[2], dims[3], 3, 2, 1)

        self.blocks1 = nn.ModuleList([TransformerBlock(dims[0], heads[0], sr_ratio=sr_ratios[0], drop=drop) for _ in range(depths[0])])
        self.blocks2 = nn.ModuleList([TransformerBlock(dims[1], heads[1], sr_ratio=sr_ratios[1], drop=drop) for _ in range(depths[1])])
        self.blocks3 = nn.ModuleList([TransformerBlock(dims[2], heads[2], sr_ratio=sr_ratios[2], drop=drop) for _ in range(depths[2])])
        self.blocks4 = nn.ModuleList([TransformerBlock(dims[3], heads[3], sr_ratio=sr_ratios[3], drop=drop) for _ in range(depths[3])])

    @staticmethod
    def run_stage(x, blocks):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in blocks:
            tokens = block(tokens, h, w)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x):
        outs = []
        x = self.patch_embed1(x)
        x = self.run_stage(x, self.blocks1)
        outs.append(x)
        x = self.patch_embed2(x)
        x = self.run_stage(x, self.blocks2)
        outs.append(x)
        x = self.patch_embed3(x)
        x = self.run_stage(x, self.blocks3)
        outs.append(x)
        x = self.patch_embed4(x)
        x = self.run_stage(x, self.blocks4)
        outs.append(x)
        return outs


class SegFormerBandBaseline(nn.Module):
    def __init__(self, decoder_dim=256, dropout=0.1, variant="b1"):
        super().__init__()
        self.backbone = MixVisionTransformerB1(drop=dropout, variant=variant)
        dims = self.backbone.embed_dims
        self.proj = nn.ModuleList([nn.Conv2d(dim, decoder_dim, 1) for dim in dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.head = nn.Conv2d(decoder_dim, 1, 1)

    def fuse_backbone_features(self, feats):
        target_size = feats[0].shape[-2:]
        projected = []
        for feat, proj in zip(feats, self.proj):
            y = proj(feat)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
            projected.append(y)
        return self.fuse(torch.cat(projected, dim=1))

    def decode_features(self, x):
        return self.fuse_backbone_features(self.backbone(x))

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.decode_features(x)
        logits = self.head(x)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


class BoundaryHead(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class SegFormerBandEdgeHead(SegFormerBandBaseline):
    def __init__(self, decoder_dim=256, dropout=0.1, variant="b1"):
        super().__init__(decoder_dim=decoder_dim, dropout=dropout, variant=variant)
        self.head = None
        self.band_head = BoundaryHead(decoder_dim, dropout=dropout)
        self.edge_head = BoundaryHead(decoder_dim, dropout=dropout)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.decode_features(x)
        return {
            "band": F.interpolate(self.band_head(features), size=input_size, mode="bilinear", align_corners=False),
            "edge": F.interpolate(self.edge_head(features), size=input_size, mode="bilinear", align_corners=False),
        }


class SegFormerBandEdgeHeatmapHead(SegFormerBandBaseline):
    """Band/edge model with an auxiliary soft centerline heatmap head."""

    def __init__(self, decoder_dim=256, dropout=0.1, variant="b1"):
        super().__init__(
            decoder_dim=decoder_dim,
            dropout=dropout,
            variant=variant,
        )
        self.head = None
        self.band_head = BoundaryHead(decoder_dim, dropout=dropout)
        self.edge_head = BoundaryHead(decoder_dim, dropout=dropout)
        self.heatmap_head = BoundaryHead(decoder_dim, dropout=dropout)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.decode_features(x)
        return {
            "band": F.interpolate(
                self.band_head(features),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "edge": F.interpolate(
                self.edge_head(features),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "heatmap": F.interpolate(
                self.heatmap_head(features),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
        }


class SegFormerBandOnlyDeployment(nn.Module):
    """Deployment view that retains only the modules required by Band output."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.proj = model.proj
        self.fuse = model.fuse
        self.band_head = getattr(model, "band_head", None)
        if self.band_head is None:
            self.band_head = model.head

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.backbone(x)
        target_size = feats[0].shape[-2:]
        projected = []
        for feat, projection in zip(feats, self.proj):
            value = projection(feat)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
        features = self.fuse(torch.cat(projected, dim=1))
        logits = self.band_head(features)
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )






















def build_model_from_config(cfg):
    name = str(cfg["model"].get("name", "segformer_b1_band_baseline")).lower()
    variant = str(cfg["model"].get("variant", "b1")).lower()
    if variant != "b1":
        raise ValueError("The public release supports the paper's B1 models only.")
    common = {
        "decoder_dim": int(cfg["model"]["decoder_dim"]),
        "dropout": float(cfg["model"].get("dropout", 0.1)),
        "variant": variant,
    }
    if name in {"segformer_b1_band_edge_heatmap", "segformer_b1_band_edge_heatmap_aux"}:
        return SegFormerBandEdgeHeatmapHead(**common)
    if name in {"segformer_b1_band_edge", "segformer_b1_band_edge_head", "segformer_b1_boundary_aux"}:
        return SegFormerBandEdgeHead(**common)
    if name == "segformer_b1_band_baseline":
        return SegFormerBandBaseline(**common)
    raise ValueError(f"Unsupported released model: {name}")


def _copy_if_shape_matches(target_state, key, value, loaded):
    if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
        target_state[key].copy_(value)
        loaded.append(key)
        return True
    return False


def _squeeze_conv1x1(weight):
    if weight.ndim == 4 and weight.shape[-2:] == (1, 1):
        return weight[:, :, 0, 0]
    return weight


def load_segformer_pretrained_backbone(model, checkpoint_path, strict_variant=True):
    checkpoint_path = Path(checkpoint_path)
    raw = torch.load(checkpoint_path, map_location="cpu")
    source_state = raw.get("state_dict") if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(source_state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    backbone = model.backbone if hasattr(model, "backbone") else model.module.backbone
    target_state = backbone.state_dict()
    loaded = []
    skipped = []

    stage_prefixes = ["1", "2", "3", "4"]
    for stage_idx, stage_no in enumerate(stage_prefixes):
        src_base = f"layers.{stage_idx}.0"
        dst_base = f"patch_embed{stage_no}"
        direct_pairs = [
            (f"{src_base}.projection.weight", f"{dst_base}.proj.weight"),
            (f"{src_base}.projection.bias", f"{dst_base}.proj.bias"),
            (f"{src_base}.norm.weight", f"{dst_base}.norm.weight"),
            (f"{src_base}.norm.bias", f"{dst_base}.norm.bias"),
        ]
        for src_key, dst_key in direct_pairs:
            if src_key in source_state:
                if not _copy_if_shape_matches(target_state, dst_key, source_state[src_key], loaded):
                    skipped.append((src_key, dst_key))

    for stage_idx, stage_no in enumerate(stage_prefixes):
        block_prefix = f"blocks{stage_no}"
        block_count = len(getattr(backbone, block_prefix))
        for block_idx in range(block_count):
            src_base = f"layers.{stage_idx}.1.{block_idx}"
            dst_base = f"{block_prefix}.{block_idx}"
            pairs = [
                (f"{src_base}.norm1.weight", f"{dst_base}.norm1.weight"),
                (f"{src_base}.norm1.bias", f"{dst_base}.norm1.bias"),
                (f"{src_base}.attn.attn.out_proj.weight", f"{dst_base}.attn.proj.weight"),
                (f"{src_base}.attn.attn.out_proj.bias", f"{dst_base}.attn.proj.bias"),
                (f"{src_base}.attn.sr.weight", f"{dst_base}.attn.sr.weight"),
                (f"{src_base}.attn.sr.bias", f"{dst_base}.attn.sr.bias"),
                (f"{src_base}.attn.norm.weight", f"{dst_base}.attn.norm.weight"),
                (f"{src_base}.attn.norm.bias", f"{dst_base}.attn.norm.bias"),
                (f"{src_base}.norm2.weight", f"{dst_base}.norm2.weight"),
                (f"{src_base}.norm2.bias", f"{dst_base}.norm2.bias"),
                (f"{src_base}.ffn.layers.0.bias", f"{dst_base}.mlp.fc1.bias"),
                (f"{src_base}.ffn.layers.1.weight", f"{dst_base}.mlp.dwconv.dwconv.weight"),
                (f"{src_base}.ffn.layers.1.bias", f"{dst_base}.mlp.dwconv.dwconv.bias"),
                (f"{src_base}.ffn.layers.4.bias", f"{dst_base}.mlp.fc2.bias"),
            ]
            for src_key, dst_key in pairs:
                if src_key in source_state:
                    src_value = source_state[src_key]
                    if src_key.endswith("ffn.layers.0.bias") or src_key.endswith("ffn.layers.4.bias"):
                        pass
                    if not _copy_if_shape_matches(target_state, dst_key, src_value, loaded):
                        skipped.append((src_key, dst_key))

            conv_linear_pairs = [
                (f"{src_base}.ffn.layers.0.weight", f"{dst_base}.mlp.fc1.weight"),
                (f"{src_base}.ffn.layers.4.weight", f"{dst_base}.mlp.fc2.weight"),
            ]
            for src_key, dst_key in conv_linear_pairs:
                if src_key in source_state:
                    src_value = _squeeze_conv1x1(source_state[src_key])
                    if not _copy_if_shape_matches(target_state, dst_key, src_value, loaded):
                        skipped.append((src_key, dst_key))

            in_proj_w_key = f"{src_base}.attn.attn.in_proj_weight"
            in_proj_b_key = f"{src_base}.attn.attn.in_proj_bias"
            if in_proj_w_key in source_state:
                in_proj_weight = source_state[in_proj_w_key]
                dim = in_proj_weight.shape[0] // 3
                q_weight = in_proj_weight[:dim]
                kv_weight = torch.cat([in_proj_weight[dim : 2 * dim], in_proj_weight[2 * dim :]], dim=0)
                if not _copy_if_shape_matches(target_state, f"{dst_base}.attn.q.weight", q_weight, loaded):
                    skipped.append((in_proj_w_key, f"{dst_base}.attn.q.weight"))
                if not _copy_if_shape_matches(target_state, f"{dst_base}.attn.kv.weight", kv_weight, loaded):
                    skipped.append((in_proj_w_key, f"{dst_base}.attn.kv.weight"))
            if in_proj_b_key in source_state:
                in_proj_bias = source_state[in_proj_b_key]
                dim = in_proj_bias.shape[0] // 3
                q_bias = in_proj_bias[:dim]
                kv_bias = torch.cat([in_proj_bias[dim : 2 * dim], in_proj_bias[2 * dim :]], dim=0)
                if not _copy_if_shape_matches(target_state, f"{dst_base}.attn.q.bias", q_bias, loaded):
                    skipped.append((in_proj_b_key, f"{dst_base}.attn.q.bias"))
                if not _copy_if_shape_matches(target_state, f"{dst_base}.attn.kv.bias", kv_bias, loaded):
                    skipped.append((in_proj_b_key, f"{dst_base}.attn.kv.bias"))

    backbone.load_state_dict(target_state, strict=True)

    source_block_keys = [k for k in source_state if ".1." in k and k.endswith("norm1.weight")]
    expected_blocks = sum(len(getattr(backbone, f"blocks{i}")) for i in range(1, 5))
    if strict_variant and len(source_block_keys) != expected_blocks:
        raise ValueError(
            f"Checkpoint block count ({len(source_block_keys)}) does not match model "
            f"variant {backbone.variant} ({expected_blocks}). Use the matching B1/B2 config."
        )

    return {
        "checkpoint": str(checkpoint_path),
        "loaded": len(loaded),
        "skipped": len(skipped),
        "loaded_keys": loaded,
        "skipped_keys": skipped[:20],
    }


def load_matching_model_weights(model, checkpoint_path, device):
    """Load checkpoint tensors that exist in the model with matching shapes."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    target_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in source_state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if (
            normalized_key in target_state
            and tuple(target_state[normalized_key].shape) == tuple(value.shape)
        ):
            compatible[normalized_key] = value
        else:
            skipped.append(key)
    load_result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded": len(compatible),
        "skipped": len(skipped),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_keys": skipped[:20],
    }


def band_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["band"]
    return outputs


def dice_loss_with_logits(logits, targets, valid=None, eps=1e-6):
    probs = torch.sigmoid(logits)
    if valid is not None:
        probs = probs * valid
        targets = targets * valid
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, targets, valid):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    dice = dice_loss_with_logits(logits, targets, valid)
    return bce + dice, bce.detach(), dice.detach()


def soft_heatmap_loss(
    logits,
    targets,
    valid,
    positive_weight=4.0,
    beta=0.1,
):
    """Regress a sparse Gaussian heatmap without letting background dominate."""
    probabilities = torch.sigmoid(logits)
    pixel_loss = F.smooth_l1_loss(
        probabilities,
        targets,
        reduction="none",
        beta=float(beta),
    )
    weights = 1.0 + float(positive_weight) * targets
    weighted_valid = weights * valid
    return (
        (pixel_loss * weighted_valid).sum()
        / weighted_valid.sum().clamp_min(1.0)
    )








def compute_loss(outputs, batch, valid, loss_weights):
    logits = outputs["band"] if isinstance(outputs, dict) else outputs
    band_loss, band_bce, band_dice = bce_dice_loss(
        logits, batch["band"], valid
    )
    total = float(loss_weights.get("band", 1.0)) * band_loss
    result = {
        "loss": total.detach(),
        "band_loss": band_loss.detach(),
        "band_bce": band_bce,
        "band_dice_loss": band_dice,
    }

    if isinstance(outputs, dict) and "edge" in outputs:
        edge_loss, edge_bce, edge_dice = bce_dice_loss(
            outputs["edge"], batch["edge"], valid
        )
        total = total + float(loss_weights.get("edge", 0.0)) * edge_loss
        result.update(
            edge_loss=edge_loss.detach(),
            edge_bce=edge_bce,
            edge_dice_loss=edge_dice,
        )

    if isinstance(outputs, dict) and "heatmap" in outputs:
        heatmap_loss = soft_heatmap_loss(
            outputs["heatmap"],
            batch["heatmap"],
            valid,
            positive_weight=float(
                loss_weights.get("heatmap_positive_weight", 4.0)
            ),
            beta=float(loss_weights.get("heatmap_beta", 0.1)),
        )
        total = total + float(loss_weights.get("heatmap", 0.0)) * heatmap_loss
        result["heatmap_loss"] = heatmap_loss.detach()

    result["loss"] = total.detach()
    return total, result




@torch.no_grad()
def compute_metrics(logits, targets, valid, threshold=0.5):
    logits = band_logits(logits)
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float() * valid
    target = (targets > 0.5).float() * valid
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    union = ((pred + target) > 0).float().sum(dims)
    pred_sum = pred.sum(dims)
    target_sum = target.sum(dims)
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
    dice = torch.where(
        (pred_sum + target_sum) > 0,
        (2 * inter) / (pred_sum + target_sum).clamp_min(1),
        torch.ones_like(inter),
    )
    return {"iou": iou.mean().item(), "dice": dice.mean().item()}


def seed_everything(seed, deterministic=False):
    seed = int(seed)
    deterministic = bool(deterministic)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Must be set before the first CUDA/cuBLAS operation in this process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def save_visualization(batch, outputs, out_dir, max_items=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    masks = batch["band"].detach().cpu()
    edge_gt = batch["edge"].detach().cpu()
    band_prob = torch.sigmoid(band_logits(outputs).detach().cpu())
    edge_prob = None
    heatmap_prob = None
    if isinstance(outputs, dict):
        edge_prob = torch.sigmoid(outputs["edge"].detach().cpu())
        if "heatmap" in outputs:
            heatmap_prob = torch.sigmoid(
                outputs["heatmap"].detach().cpu()
            )
    valid = batch["valid"].detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(min(max_items, images.shape[0])):
        image = (images[i] * std + mean).clamp(0, 1)
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = (masks[i, 0].numpy() * 255).astype(np.uint8)
        pred = (band_prob[i, 0].numpy() * valid[i, 0].numpy() * 255).astype(np.uint8)
        overlay = image_np.copy()
        overlay[pred > 127] = (0.45 * overlay[pred > 127] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        overlay[gt > 127] = (0.45 * overlay[gt > 127] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        panels = [
            image_np,
            cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB),
            overlay,
        ]
        if edge_prob is not None:
            edge_gt_u8 = (edge_gt[i, 0].numpy() * 255).astype(np.uint8)
            edge_pred_u8 = (edge_prob[i, 0].numpy() * valid[i, 0].numpy() * 255).astype(np.uint8)
            panels.extend(
                [
                    cv2.cvtColor(edge_gt_u8, cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(edge_pred_u8, cv2.COLOR_GRAY2RGB),
                ]
            )
        if heatmap_prob is not None and "heatmap" in batch:
            heatmap_gt_u8 = (
                batch["heatmap"][i, 0].detach().cpu().numpy() * 255
            ).astype(np.uint8)
            heatmap_pred_u8 = (
                heatmap_prob[i, 0].numpy()
                * valid[i, 0].numpy()
                * 255
            ).astype(np.uint8)
            panels.extend(
                [
                    cv2.cvtColor(heatmap_gt_u8, cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(heatmap_pred_u8, cv2.COLOR_GRAY2RGB),
                ]
            )
        panel = np.concatenate(panels, axis=1)
        Image.fromarray(panel).save(out_dir / f"{batch['id'][i]}_vis.jpg")


def average_totals(totals, steps):
    return {k: float(v) / max(steps, 1) for k, v in totals.items()}


def add_loss_items(totals, loss_items):
    for key, value in loss_items.items():
        totals[key] = totals.get(key, 0.0) + float(value.item())


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled, epoch, log_interval, max_steps, loss_weights):
    model.train()
    totals = {}
    steps = 0
    pbar = tqdm(loader, desc=f"train {epoch}", ncols=100)
    for step, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        for key in (
            "band",
            "edge",
            "heatmap",
        ):
            if key not in batch:
                continue
            batch[key] = batch[key].to(device, non_blocking=True)
        batch["has_boundary"] = batch["has_boundary"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss, loss_items = compute_loss(outputs, batch, valid, loss_weights)
        if not torch.isfinite(loss):
            values = {
                key: float(value.detach().cpu())
                for key, value in loss_items.items()
            }
            raise FloatingPointError(
                f"Non-finite training loss at epoch={epoch}, step={step}: "
                f"{values}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        add_loss_items(totals, loss_items)
        steps += 1
        if step % log_interval == 0:
            pbar.set_postfix(loss=totals["loss"] / steps)
        if max_steps and step >= max_steps:
            break
    return average_totals(totals, steps)


@torch.no_grad()
def validate(model, loader, device, amp_enabled, max_steps, loss_weights, vis_dir=None):
    model.eval()
    totals = {"iou": 0.0, "dice": 0.0}
    steps = 0
    first_batch = None
    first_logits = None
    for step, batch in enumerate(tqdm(loader, desc="val", ncols=100), start=1):
        images = batch["image"].to(device, non_blocking=True)
        for key in (
            "band",
            "edge",
            "heatmap",
        ):
            if key not in batch:
                continue
            batch[key] = batch[key].to(device, non_blocking=True)
        batch["has_boundary"] = batch["has_boundary"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            _, loss_items = compute_loss(outputs, batch, valid, loss_weights)
        if not all(torch.isfinite(value) for value in loss_items.values()):
            values = {
                key: float(value.detach().cpu())
                for key, value in loss_items.items()
            }
            raise FloatingPointError(
                f"Non-finite validation loss at step={step}: {values}"
            )
        metrics = compute_metrics(outputs, batch["band"], valid)
        add_loss_items(totals, loss_items)
        totals["iou"] += metrics["iou"]
        totals["dice"] += metrics["dice"]
        steps += 1
        if first_batch is None:
            first_batch = batch
            first_logits = outputs
        if max_steps and step >= max_steps:
            break
    if vis_dir is not None and first_batch is not None:
        save_visualization(first_batch, first_logits, vis_dir)
    return average_totals(totals, steps)


@torch.no_grad()
def evaluate_test_split(
    model,
    loader,
    device,
    amp_enabled,
    loss_weights,
    threshold=0.5,
    vis_dir=None,
):
    """Evaluate the held-out test split once after model selection is complete."""
    model.eval()
    loss_totals = {}
    all_iou = []
    all_dice = []
    positive_iou = []
    positive_dice = []
    positive_zero_predictions = 0
    positive_samples = 0
    negative_false_positives = 0
    negative_samples = 0
    first_batch = None
    first_outputs = None

    for batch in tqdm(loader, desc="test(best)", ncols=100):
        images = batch["image"].to(device, non_blocking=True)
        for key in (
            "band",
            "edge",
            "heatmap",
        ):
            if key not in batch:
                continue
            batch[key] = batch[key].to(device, non_blocking=True)
        batch["has_boundary"] = batch["has_boundary"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            _, loss_items = compute_loss(outputs, batch, valid, loss_weights)
        batch_size = images.shape[0]
        for key, value in loss_items.items():
            loss_totals[key] = loss_totals.get(key, 0.0) + float(value.item()) * batch_size

        probs = torch.sigmoid(band_logits(outputs))
        pred = (probs > threshold).float() * valid
        target = (batch["band"] > 0.5).float() * valid
        dims = (1, 2, 3)
        inter = (pred * target).sum(dims)
        union = ((pred + target) > 0).float().sum(dims)
        pred_sum = pred.sum(dims)
        target_sum = target.sum(dims)
        iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
        dice = torch.where(
            (pred_sum + target_sum) > 0,
            (2 * inter) / (pred_sum + target_sum).clamp_min(1),
            torch.ones_like(inter),
        )

        for index in range(batch_size):
            sample_iou = float(iou[index].item())
            sample_dice = float(dice[index].item())
            all_iou.append(sample_iou)
            all_dice.append(sample_dice)
            if float(target_sum[index].item()) > 0:
                positive_samples += 1
                positive_iou.append(sample_iou)
                positive_dice.append(sample_dice)
                if float(pred_sum[index].item()) <= 0:
                    positive_zero_predictions += 1
            else:
                negative_samples += 1
                if float(pred_sum[index].item()) > 0:
                    negative_false_positives += 1

        if first_batch is None:
            first_batch = batch
            first_outputs = outputs

    sample_count = len(all_iou)
    metrics = {
        "samples": sample_count,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "iou_all": float(np.mean(all_iou)) if all_iou else 0.0,
        "dice_all": float(np.mean(all_dice)) if all_dice else 0.0,
        "iou_positive": float(np.mean(positive_iou)) if positive_iou else 0.0,
        "dice_positive": float(np.mean(positive_dice)) if positive_dice else 0.0,
        "positive_zero_prediction_rate": (
            positive_zero_predictions / positive_samples if positive_samples else 0.0
        ),
        "negative_false_positive_rate": (
            negative_false_positives / negative_samples if negative_samples else 0.0
        ),
    }
    metrics.update({key: value / max(sample_count, 1) for key, value in loss_totals.items()})

    if vis_dir is not None and first_batch is not None:
        save_visualization(first_batch, first_outputs, vis_dir)
    return metrics


def write_single_metrics_row(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        raise ValueError("Checkpoint does not contain RNG state.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda_all"]]
        )


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_checkpoint_payload(
    model,
    optimizer,
    scaler,
    train_sampler,
    epoch,
    metrics,
    best_iou,
    best_epoch,
    no_improve_epochs,
):
    sampler_state = None
    if train_sampler is not None and getattr(train_sampler, "generator", None) is not None:
        sampler_state = train_sampler.generator.get_state()
    return {
        "checkpoint_version": 2,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp_scaler": scaler.state_dict(),
        "metrics": metrics,
        "training_state": {
            "best_iou": float(best_iou),
            "best_epoch": int(best_epoch),
            "no_improve_epochs": int(no_improve_epochs),
        },
        "rng_state": capture_rng_state(),
        "sampler_generator_state": sampler_state,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    train_sampler,
    epoch,
    metrics,
    best_iou,
    best_epoch,
    no_improve_epochs,
):
    payload = build_checkpoint_payload(
        model,
        optimizer,
        scaler,
        train_sampler,
        epoch,
        metrics,
        best_iou,
        best_epoch,
        no_improve_epochs,
    )
    atomic_torch_save(payload, path)
    return payload


def trim_metrics_after_epoch(path, last_complete_epoch):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = [
            row
            for row in reader
            if int(row.get("epoch", 0)) <= int(last_complete_epoch)
        ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def append_metrics(path, epoch, train_metrics, val_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = ["epoch"] + [f"train_{k}" for k in train_metrics] + [f"val_{k}" for k in val_metrics]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer-B1 style baseline for emergency outer boundary.")
    parser.add_argument("--config", default="configs/main/a0_band_only.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training, augmentation, and sampler random seed.",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume training from.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Load model weights only and start a fresh fine-tuning run with a "
            "new optimizer and epoch counter."
        ),
    )
    parser.add_argument(
        "--init-checkpoint-partial",
        default=None,
        help=(
            "Load matching model tensors only, leaving newly added modules at "
            "their configured initialization."
        ),
    )
    parser.add_argument("--pretrained-backbone", default=None, help="SegFormer B1/B2 backbone checkpoint to load before training.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop if val_iou does not improve for this many consecutive epochs. 0 disables early stopping.",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Do not evaluate the held-out test split after training finishes.",
    )
    args = parser.parse_args()
    initialization_modes = [
        bool(args.resume),
        bool(args.init_checkpoint),
        bool(args.init_checkpoint_partial),
    ]
    if sum(initialization_modes) > 1:
        parser.error(
            "--resume, --init-checkpoint and --init-checkpoint-partial "
            "are mutually exclusive."
        )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.max_train_steps is not None:
        train_cfg["max_train_steps"] = args.max_train_steps
    if args.max_val_steps is not None:
        train_cfg["max_val_steps"] = args.max_val_steps
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.early_stop_patience is not None:
        train_cfg["early_stop_patience"] = args.early_stop_patience

    deterministic = bool(train_cfg.get("deterministic", False))
    seed_everything(
        int(train_cfg.get("seed", 42)),
        deterministic=deterministic,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    dataset_cfg = cfg["dataset"]
    train_set = BoundaryDataset(
        dataset_cfg["root"],
        dataset_cfg["train_csv"],
        dataset_cfg["image_size"],
        dataset_cfg["image_mean"],
        dataset_cfg["image_std"],
        train=True,
        augmentation=train_cfg.get("augmentation", "basic"),
    )
    val_set = BoundaryDataset(
        dataset_cfg["root"],
        dataset_cfg["val_csv"],
        dataset_cfg["image_size"],
        dataset_cfg["image_mean"],
        dataset_cfg["image_std"],
        train=False,
    )
    evaluate_test_after_training = bool(train_cfg.get("evaluate_test_after_training", True))
    test_set = None
    if evaluate_test_after_training and not args.skip_final_test and dataset_cfg.get("test_csv"):
        test_set = BoundaryDataset(
            dataset_cfg["root"],
            dataset_cfg["test_csv"],
            dataset_cfg["image_size"],
            dataset_cfg["image_mean"],
            dataset_cfg["image_std"],
            train=False,
        )
    train_sampler, sampler_stats = build_train_sampler(
        train_set,
        train_cfg.get("sampling", {}),
        int(train_cfg.get("seed", 42)),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model = build_model_from_config(cfg).to(device)
    pretrained_path = args.pretrained_backbone or cfg["model"].get("pretrained_backbone")
    if (
        pretrained_path
        and not args.resume
        and not args.init_checkpoint
        and not args.init_checkpoint_partial
    ):
        load_info = load_segformer_pretrained_backbone(model, pretrained_path)
        print(
            f"Loaded pretrained backbone: {load_info['checkpoint']} | "
            f"loaded={load_info['loaded']} skipped={load_info['skipped']}"
        )
        if load_info["skipped_keys"]:
            print(f"Skipped examples: {load_info['skipped_keys']}")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        print(
            "Initialized model weights for fresh fine-tuning: "
            f"{Path(args.init_checkpoint).resolve()}"
        )
    if args.init_checkpoint_partial:
        load_info = load_matching_model_weights(
            model,
            args.init_checkpoint_partial,
            device,
        )
        print(
            "Partially initialized model weights for fresh fine-tuning: "
            f"{Path(args.init_checkpoint_partial).resolve()} | "
            f"loaded={load_info['loaded']} skipped={load_info['skipped']} "
            f"new_or_missing={len(load_info['missing_keys'])}"
        )
        if load_info["skipped_keys"]:
            print(f"Skipped examples: {load_info['skipped_keys']}")
    loss_weights = cfg.get("loss", {"band": 1.0, "edge": 1.0})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir = Path(train_cfg["output_dir"])
    if deterministic and output_dir.exists() and not args.resume:
        raise FileExistsError(
            "Deterministic registered run refuses an existing output directory. "
            f"Use a new directory or the complete-state --resume path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    metrics_path = output_dir / "metrics.csv"
    best_iou = -math.inf
    start_epoch = 1
    best_epoch = 0
    no_improve_epochs = 0
    resume_rng_state = None
    resume_sampler_state = None

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        if deterministic and int(checkpoint.get("checkpoint_version", 0)) < 2:
            raise ValueError(
                "Strict deterministic resume requires a version-2 complete-state "
                "checkpoint created by this experiment."
            )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "amp_scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["amp_scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        metric_rows = trim_metrics_after_epoch(
            metrics_path,
            int(checkpoint.get("epoch", 0)),
        )
        for row in metric_rows:
            try:
                val_iou = float(row.get("val_iou", "-inf"))
                if val_iou > best_iou:
                    best_iou = val_iou
                    best_epoch = int(row.get("epoch", 0))
            except ValueError:
                pass
        if best_iou == -math.inf:
            training_state = checkpoint.get("training_state", {})
            best_iou = float(
                training_state.get(
                    "best_iou",
                    checkpoint.get("metrics", {}).get("iou", -math.inf),
                )
            )
            best_epoch = int(
                training_state.get(
                    "best_epoch",
                    checkpoint.get("epoch", 0),
                )
            )
        no_improve_epochs = max(
            0,
            int(checkpoint.get("epoch", 0)) - int(best_epoch),
        )
        resume_rng_state = checkpoint.get("rng_state")
        resume_sampler_state = checkpoint.get("sampler_generator_state")

        best_checkpoint_path = output_dir / "checkpoints" / "best.pt"
        if best_epoch == int(checkpoint.get("epoch", 0)):
            atomic_torch_save(checkpoint, best_checkpoint_path)
        elif not best_checkpoint_path.exists() and deterministic:
            raise FileNotFoundError(
                "Cannot reconstruct the registered best checkpoint during resume: "
                f"{best_checkpoint_path}"
            )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    sample_message = f"Train samples: {len(train_set)} | Val samples: {len(val_set)}"
    if test_set is not None:
        sample_message += f" | Test samples: {len(test_set)} (evaluated once after training)"
    print(sample_message)
    if sampler_stats is None:
        print("Train sampling: random shuffle")
    else:
        print(f"Train sampling: {sampler_stats}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {int(train_cfg.get('seed', 42))}")
    print(
        "Deterministic mode: "
        f"{deterministic} | "
        f"algorithms={torch.are_deterministic_algorithms_enabled()} | "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark} | "
        f"cudnn_deterministic={torch.backends.cudnn.deterministic} | "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}"
    )
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
    if early_stop_patience > 0:
        print(f"Early stopping: enabled | metric=val_iou | patience={early_stop_patience}")
    if args.resume:
        if resume_rng_state is None:
            raise ValueError("Resume checkpoint is missing complete RNG state.")
        restore_rng_state(resume_rng_state)
        if train_sampler is not None:
            if resume_sampler_state is None:
                raise ValueError(
                    "Resume checkpoint is missing sampler generator state."
                )
            train_sampler.generator.set_state(resume_sampler_state.cpu())
        print(
            "Complete-state resume restored: "
            f"start_epoch={start_epoch} best_epoch={best_epoch} "
            f"best_iou={best_iou:.6f} no_improve={no_improve_epochs}"
        )
        if (
            early_stop_patience > 0
            and no_improve_epochs >= early_stop_patience
        ):
            print(
                "Resume checkpoint had already satisfied early stopping; "
                "no additional epochs will be trained."
            )
            return

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            epoch,
            int(train_cfg.get("log_interval", 20)),
            int(train_cfg.get("max_train_steps", 0)),
            loss_weights,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            amp_enabled,
            int(train_cfg.get("max_val_steps", 0)),
            loss_weights,
            output_dir / "visualizations" / f"epoch_{epoch:03d}",
        )
        model_for_metrics = model.module if hasattr(model, "module") else model
        if hasattr(model_for_metrics, "residual_alpha"):
            train_metrics["residual_alpha"] = float(
                model_for_metrics.residual_alpha().detach().cpu()
            )
        append_metrics(metrics_path, epoch, train_metrics, val_metrics)
        print(f"Epoch {epoch}: train={train_metrics} val={val_metrics}")

        improved = val_metrics["iou"] > best_iou
        if improved:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        last_payload = save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scaler,
            train_sampler,
            epoch,
            val_metrics,
            best_iou,
            best_epoch,
            no_improve_epochs,
        )
        if improved:
            atomic_torch_save(
                last_payload,
                output_dir / "checkpoints" / "best.pt",
            )
        if int(train_cfg.get("save_interval", 5)) and epoch % int(train_cfg.get("save_interval", 5)) == 0:
            atomic_torch_save(
                last_payload,
                output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
            )
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: val_iou has not improved for "
                f"{no_improve_epochs} epochs. Best epoch={best_epoch}, best val_iou={best_iou:.6f}."
            )
            break

    if test_loader is not None:
        best_checkpoint_path = output_dir / "checkpoints" / "best.pt"
        if not best_checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot run final test because best checkpoint is missing: {best_checkpoint_path}")
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model"])
        selected_epoch = int(best_checkpoint.get("epoch", best_epoch))
        print(f"\nFinal held-out test: loading {best_checkpoint_path} (epoch {selected_epoch})")
        test_metrics = evaluate_test_split(
            model,
            test_loader,
            device,
            amp_enabled,
            loss_weights,
            threshold=float(train_cfg.get("test_threshold", 0.5)),
            vis_dir=output_dir / "visualizations" / "final_test_best",
        )
        test_metrics = {"checkpoint_epoch": selected_epoch, **test_metrics}
        write_single_metrics_row(output_dir / "test_metrics.csv", test_metrics)
        print("Final test metrics (test was not used for checkpoint selection):")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"Saved: {output_dir / 'test_metrics.csv'}")


if __name__ == "__main__":
    main()
