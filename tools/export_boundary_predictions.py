import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_emergency_boundary_baseline import BoundaryDataset, band_logits, build_model_from_config, compute_metrics


def load_prefix_metadata(path):
    metadata = {}
    if not path:
        return metadata
    path = Path(path)
    if not path.exists():
        return metadata
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metadata[str(row["prefix"])] = row
    return metadata


def denormalize(image_t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image_t.cpu() * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_panel(image_np, gt, pred_prob, valid, threshold, extra_probs=None):
    pred = ((pred_prob >= threshold).astype(np.uint8) * valid.astype(np.uint8)) * 255
    prob_u8 = np.clip(pred_prob * valid * 255, 0, 255).astype(np.uint8)
    gt_u8 = (gt.astype(np.uint8) * valid.astype(np.uint8)) * 255

    overlay = image_np.copy()
    overlay[prob_u8 > 127] = (0.45 * overlay[prob_u8 > 127] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[gt_u8 > 127] = (0.45 * overlay[gt_u8 > 127] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)

    panels = [
        image_np,
        cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(prob_u8, cv2.COLOR_GRAY2RGB),
        overlay,
    ]
    if extra_probs:
        for prob in extra_probs:
            panels.append(cv2.cvtColor(np.clip(prob * valid * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB))
    panel = np.concatenate(panels, axis=1)
    return panel


def per_sample_metrics(prob, gt, valid, threshold):
    pred = (prob >= threshold).astype(np.float32) * valid
    target = (gt > 0.5).astype(np.float32) * valid
    inter = float((pred * target).sum())
    union = float(((pred + target) > 0).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    if union > 0:
        iou = inter / union
    else:
        iou = 1.0
    if pred_sum + target_sum > 0:
        dice = 2.0 * inter / (pred_sum + target_sum)
    else:
        dice = 1.0
    return iou, dice, pred_sum, target_sum


def selected_gt_from_batch(batch, prediction_head):
    if prediction_head == "edge":
        return batch["edge"]
    return batch["mask"]


def build_group_metrics(rows, group_key):
    stats = {}
    for row in rows:
        key = row.get(group_key) or "unknown"
        item = stats.setdefault(
            key,
            {
                "count": 0,
                "with_boundary": 0,
                "iou": 0.0,
                "dice": 0.0,
                "pos_count": 0,
                "pos_iou": 0.0,
                "pos_dice": 0.0,
                "zero_pred_pos": 0,
                "neg_count": 0,
                "false_positive_neg": 0,
            },
        )
        has_boundary = int(row["has_boundary"])
        pred_pixels = float(row["pred_pixels"])
        iou = float(row["iou"])
        dice = float(row["dice"])

        item["count"] += 1
        item["with_boundary"] += has_boundary
        item["iou"] += iou
        item["dice"] += dice
        if has_boundary:
            item["pos_count"] += 1
            item["pos_iou"] += iou
            item["pos_dice"] += dice
            if pred_pixels <= 0:
                item["zero_pred_pos"] += 1
        else:
            item["neg_count"] += 1
            if pred_pixels > 0:
                item["false_positive_neg"] += 1
    return stats


def write_group_metrics(path, stats, group_key):
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            group_key,
            "count",
            "with_boundary",
            "positive_count",
            "negative_count",
            "mean_iou_all",
            "mean_dice_all",
            "mean_iou_positive",
            "mean_dice_positive",
            "positive_zero_prediction_rate",
            "negative_false_positive_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, item in sorted(stats.items()):
            pos_count = item["pos_count"]
            neg_count = item["neg_count"]
            writer.writerow(
                {
                    group_key: key,
                    "count": item["count"],
                    "with_boundary": item["with_boundary"],
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "mean_iou_all": item["iou"] / item["count"] if item["count"] else "",
                    "mean_dice_all": item["dice"] / item["count"] if item["count"] else "",
                    "mean_iou_positive": item["pos_iou"] / pos_count if pos_count else "",
                    "mean_dice_positive": item["pos_dice"] / pos_count if pos_count else "",
                    "positive_zero_prediction_rate": item["zero_pred_pos"] / pos_count if pos_count else "",
                    "negative_false_positive_rate": item["false_positive_neg"] / neg_count if neg_count else "",
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Export full split visualizations and per-sample metrics.")
    parser.add_argument("--config", default="configs/main/a0_band_only.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "test_seen", "test_unseen"],
        default="val",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prediction-head", choices=["band", "edge"], default="band")
    parser.add_argument("--metadata", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    split_csv = f"splits/{args.split}.csv"
    dataset = BoundaryDataset(
        dataset_cfg["root"],
        split_csv,
        dataset_cfg["image_size"],
        dataset_cfg["image_mean"],
        dataset_cfg["image_std"],
        train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prefix_metadata = load_prefix_metadata(args.metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device)
    # Training checkpoints include optimizer/RNG metadata in addition to tensors.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations"
    pred_dir = out_dir / "pred_masks"
    prob_dir = out_dir / "probability_masks"
    edge_dir = out_dir / "edge_probs"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "mask_export_protocol.json").write_text(
        json.dumps(
            {
                "protocol_version": "binary_mask_v2_clean601",
                "probability_resize": "model logits are bilinearly resized to input size before sigmoid",
                "binary_rule": f"probability >= {args.threshold}",
                "binary_values": [0, 255],
                "evaluation_rule": "saved_mask >= 128",
                "probability_archive": "probability_masks/*_prob.png uses floor clip(probability * 255)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    totals = {"iou": 0.0, "dice": 0.0}
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"export {args.split}", ncols=100):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            valid = batch["valid"].to(device)
            selected_targets = selected_gt_from_batch(batch, args.prediction_head).to(device)
            outputs = model(images)
            band_prob = torch.sigmoid(band_logits(outputs)).cpu().numpy()
            edge_prob = None
            presence_prob = None
            if isinstance(outputs, dict):
                if "edge" in outputs:
                    edge_prob = torch.sigmoid(outputs["edge"]).cpu().numpy()
                if "presence" in outputs:
                    presence_prob = torch.sigmoid(outputs["presence"]).cpu().numpy()
                if args.prediction_head == "edge":
                    if edge_prob is None:
                        raise ValueError("prediction-head=edge requested, but model has no edge output.")
                    probs = edge_prob
                else:
                    probs = band_prob
            else:
                probs = band_prob
            masks_np = selected_targets.cpu().numpy()
            valid_np = valid.cpu().numpy()

            for i, sample_id in enumerate(batch["id"]):
                prefix = str(sample_id).split("_")[0]
                meta = prefix_metadata.get(prefix, {})
                image_np = denormalize(batch["image"][i])
                gt = masks_np[i, 0]
                prob = probs[i, 0]
                valid_i = valid_np[i, 0]
                iou, dice, pred_pixels, gt_pixels = per_sample_metrics(prob, gt, valid_i, args.threshold)
                extras = []
                if edge_prob is not None:
                    extras.append(edge_prob[i, 0])
                panel = make_panel(image_np, gt, prob, valid_i, args.threshold, extra_probs=extras)
                Image.fromarray(panel).save(vis_dir / f"{sample_id}_vis.jpg")
                probability_u8 = np.clip(prob * valid_i * 255, 0, 255).astype(np.uint8)
                binary_u8 = (
                    (prob >= args.threshold).astype(np.uint8)
                    * valid_i.astype(np.uint8)
                    * 255
                )
                Image.fromarray(binary_u8).save(pred_dir / f"{sample_id}.png")
                Image.fromarray(probability_u8).save(prob_dir / f"{sample_id}_prob.png")
                if edge_prob is not None:
                    Image.fromarray(np.clip(edge_prob[i, 0] * valid_i * 255, 0, 255).astype(np.uint8)).save(
                        edge_dir / f"{sample_id}_edge.png"
                    )
                rows.append(
                    {
                        "id": sample_id,
                        "prefix": prefix,
                        "scene_category": meta.get("scene_category", "unknown"),
                        "difficulty": meta.get("difficulty", "unknown"),
                        "role": meta.get("role", "unknown"),
                        "has_boundary": int(batch["has_boundary"][i].item()),
                        "presence_score": float(presence_prob[i]) if presence_prob is not None else "",
                        "iou": iou,
                        "dice": dice,
                        "pred_pixels": pred_pixels,
                        "gt_pixels": gt_pixels,
                    }
                )
                totals["iou"] += iou
                totals["dice"] += dice
                count += 1

    metrics_path = out_dir / "per_sample_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "prefix",
            "scene_category",
            "difficulty",
            "role",
            "has_boundary",
            "presence_score",
            "iou",
            "dice",
            "pred_pixels",
            "gt_pixels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    prefix_path = out_dir / "prefix_metrics.csv"
    scene_path = out_dir / "scene_metrics.csv"
    write_group_metrics(prefix_path, build_group_metrics(rows, "prefix"), "prefix")
    write_group_metrics(scene_path, build_group_metrics(rows, "scene_category"), "scene_category")

    print(f"Exported {len(rows)} samples to {out_dir}")
    print(f"Mean IoU: {totals['iou'] / max(count, 1):.4f}")
    print(f"Mean Dice: {totals['dice'] / max(count, 1):.4f}")
    print(f"Per-sample metrics: {metrics_path}")
    print(f"Prefix metrics: {prefix_path}")
    print(f"Scene metrics: {scene_path}")


if __name__ == "__main__":
    main()
