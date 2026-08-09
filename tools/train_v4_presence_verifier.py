"""Train V4 semantic presence gating without changing the boundary model."""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.v4_presence_verifier import V4PresenceVerifier
from train_emergency_boundary_baseline import (
    BoundaryDataset,
    build_model_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-backbone semantic verifier for V4 output gating."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/system/v4_presence_verifier.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def scene_key(row, group_fields):
    for field in group_fields:
        value = str(row.get(field, "")).strip()
        if value:
            return f"{field}:{value}"
    return f"sample:{row.get('sample_id', row.get('image', 'unknown'))}"


def build_balanced_scene_sampler(dataset, group_fields, seed):
    labels = [int(row["has_boundary"]) for row in dataset.rows]
    groups = [scene_key(row, group_fields) for row in dataset.rows]
    group_counts = Counter(zip(labels, groups))

    raw_weights = np.asarray(
        [
            1.0 / math.sqrt(group_counts[(label, group)])
            for label, group in zip(labels, groups)
        ],
        dtype=np.float64,
    )
    labels_np = np.asarray(labels, dtype=np.int64)
    for label in (0, 1):
        selected = labels_np == label
        class_sum = float(raw_weights[selected].sum())
        if class_sum <= 0:
            raise ValueError(f"Training split has no class {label} samples.")
        raw_weights[selected] *= 0.5 / class_sum

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        torch.from_numpy(raw_weights).double(),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return sampler, {
        "positive_samples": int(labels_np.sum()),
        "negative_samples": int((labels_np == 0).sum()),
        "positive_scenes": len({g for y, g in zip(labels, groups) if y == 1}),
        "negative_scenes": len({g for y, g in zip(labels, groups) if y == 0}),
        "epoch_samples": len(dataset),
        "replacement": True,
    }


def load_frozen_boundary_model(config_path, checkpoint_path, device):
    with Path(config_path).open("r", encoding="utf-8") as stream:
        boundary_cfg = yaml.safe_load(stream)
    model = build_model_from_config(boundary_cfg).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, boundary_cfg


def confusion_metrics(targets, probabilities, threshold):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= float(threshold)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(targets), 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(targets, probabilities):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = probabilities[targets == 1]
    negative = probabilities[targets == 0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.0
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
        / comparisons.size
    )


def select_threshold(targets, probabilities, min_specificity):
    candidates = np.linspace(0.02, 0.98, 97)
    rows = [
        confusion_metrics(targets, probabilities, threshold)
        for threshold in candidates
    ]
    feasible = [
        row for row in rows if row["specificity"] >= float(min_specificity)
    ]
    if feasible:
        return max(
            feasible,
            key=lambda row: (
                row["recall"],
                row["balanced_accuracy"],
                -row["threshold"],
            ),
        ), "specificity_constrained"
    return max(
        rows,
        key=lambda row: (row["balanced_accuracy"], row["recall"]),
    ), "max_balanced_accuracy_fallback"


def forward_presence(boundary_model, verifier, images, train_classifier):
    with torch.no_grad():
        f4 = boundary_model.backbone(images)[-1]
    logits = verifier(f4.detach())
    if not train_classifier:
        logits = logits.detach()
    return logits


def run_epoch(
    loader,
    boundary_model,
    verifier,
    device,
    optimizer=None,
    max_steps=0,
    label_smoothing=0.0,
):
    training = optimizer is not None
    verifier.train(training)
    losses = []
    targets = []
    probabilities = []
    description = "train-v4" if training else "val-v4"

    for step, batch in enumerate(tqdm(loader, desc=description, ncols=100), 1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["has_boundary"].float().to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = forward_presence(
            boundary_model,
            verifier,
            images,
            train_classifier=training,
        )
        loss_targets = labels
        if training and label_smoothing > 0.0:
            loss_targets = (
                labels * (1.0 - label_smoothing)
                + 0.5 * label_smoothing
            )
        loss = F.binary_cross_entropy_with_logits(logits, loss_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite V4 loss at step {step}: {loss}")
        if training:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        targets.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
        probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
        if max_steps > 0 and step >= max_steps:
            break

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "targets": targets,
        "probabilities": probabilities,
    }


def write_history(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    train_cfg = cfg["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir

    output_dir = Path(train_cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)

    boundary_model, boundary_cfg = load_frozen_boundary_model(
        cfg["boundary_model"]["config"],
        cfg["boundary_model"]["checkpoint"],
        device,
    )
    dataset_cfg = boundary_cfg["dataset"]
    expected_size = tuple(dataset_cfg["image_size"])
    if expected_size != (450, 800):
        raise ValueError(
            f"V4 requires [450, 800], got {list(expected_size)}"
        )
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
    group_fields = train_cfg.get(
        "group_fields",
        ["reviewed_segment_id", "capture_session", "prefix"],
    )
    sampler, sampler_stats = build_balanced_scene_sampler(
        train_set,
        group_fields,
        seed,
    )
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 16)),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    feature_channels = int(boundary_model.backbone.embed_dims[-1])
    verifier = V4PresenceVerifier(
        feature_channels=feature_channels,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        dropout=float(cfg["model"].get("dropout", 0.30)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        verifier.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
    )

    history = []
    best_score = -1.0
    bad_epochs = 0
    patience = int(train_cfg.get("early_stop_patience", 8))
    min_specificity = float(train_cfg.get("min_val_specificity", 0.90))
    epochs = int(train_cfg.get("epochs", 30))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(
            f"label_smoothing must be in [0, 1), got {label_smoothing}"
        )

    print(f"DEVICE={device}")
    print(f"BOUNDARY_MODEL=FROZEN")
    print(f"BOUNDARY_CONFIG={Path(cfg['boundary_model']['config']).resolve()}")
    print(f"BOUNDARY_CHECKPOINT={Path(cfg['boundary_model']['checkpoint']).resolve()}")
    print(f"TRAIN={len(train_set)} VAL={len(val_set)}")
    print(f"SAMPLER={json.dumps(sampler_stats, ensure_ascii=False)}")

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(
            train_loader,
            boundary_model,
            verifier,
            device,
            optimizer=optimizer,
            max_steps=args.max_train_steps,
            label_smoothing=label_smoothing,
        )
        val_result = run_epoch(
            val_loader,
            boundary_model,
            verifier,
            device,
            max_steps=args.max_val_steps,
            label_smoothing=0.0,
        )
        threshold_metrics, threshold_mode = select_threshold(
            val_result["targets"],
            val_result["probabilities"],
            min_specificity,
        )
        auc = roc_auc(
            val_result["targets"],
            val_result["probabilities"],
        )
        selection_score = float(threshold_metrics["balanced_accuracy"])
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_auc": auc,
            "val_threshold": threshold_metrics["threshold"],
            "val_accuracy": threshold_metrics["accuracy"],
            "val_recall": threshold_metrics["recall"],
            "val_specificity": threshold_metrics["specificity"],
            "val_balanced_accuracy": threshold_metrics["balanced_accuracy"],
            "threshold_mode": threshold_mode,
        }
        history.append(row)
        write_history(output_dir / "metrics.csv", history)
        print(
            f"Epoch {epoch}: train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} auc={auc:.6f} "
            f"threshold={row['val_threshold']:.2f} "
            f"recall={row['val_recall']:.4f} "
            f"specificity={row['val_specificity']:.4f} "
            f"balanced_acc={selection_score:.4f}"
        )

        if selection_score > best_score + 1e-8:
            best_score = selection_score
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": verifier.state_dict(),
                    "model_config": {
                        "feature_channels": feature_channels,
                        "hidden_dim": verifier.hidden_dim,
                        "dropout": verifier.dropout,
                        "pool_regions": [
                            "global",
                            "right_half",
                            "lower_right",
                        ],
                    },
                    "presence_threshold": threshold_metrics["threshold"],
                    "threshold_mode": threshold_mode,
                    "val_metrics": {**threshold_metrics, "auc": auc},
                    "boundary_config": str(
                        Path(cfg["boundary_model"]["config"]).resolve()
                    ),
                    "boundary_checkpoint": str(
                        Path(cfg["boundary_model"]["checkpoint"]).resolve()
                    ),
                    "image_size": list(expected_size),
                    "seed": seed,
                    "label_smoothing": label_smoothing,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            bad_epochs += 1
        if patience > 0 and bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best score={best_score:.6f}")
            break

    summary = {
        "best_val_balanced_accuracy": best_score,
        "epochs_completed": len(history),
        "sampler": sampler_stats,
        "test_used_for_selection": False,
        "checkpoint": str((checkpoint_dir / "best.pt").resolve()),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"BEST_CHECKPOINT={(checkpoint_dir / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
