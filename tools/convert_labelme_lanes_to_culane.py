import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


LANE_LABELS = {"lane1": 1, "lane2": 2, "lane3": 3, "lane4": 4}


def read_labelme(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_lane_points(data):
    lanes = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in LANE_LABELS:
            continue
        points = shape.get("points", [])
        if not points:
            continue
        lane_id = LANE_LABELS[label]
        lanes.setdefault(lane_id, [])
        lanes[lane_id].extend([[float(x), float(y)] for x, y in points])

    cleaned = {}
    for lane_id, points in lanes.items():
        points = sorted(points, key=lambda p: p[1])
        deduped = []
        for x, y in points:
            if deduped and abs(deduped[-1][1] - y) < 1e-3:
                deduped[-1][0] = (deduped[-1][0] + x) / 2.0
            else:
                deduped.append([x, y])
        if len(deduped) >= 2:
            cleaned[lane_id] = deduped
    return cleaned


def draw_lane_mask(height, width, lanes, thickness):
    mask = np.zeros((height, width), dtype=np.uint8)
    for lane_id, points in lanes.items():
        pts = np.asarray(points, dtype=np.float32)
        y = pts[:, 1]
        x = pts[:, 0]
        order = np.argsort(y)
        y = y[order]
        x = x[order]

        if len(points) >= 4 and y[-1] > y[0]:
            try:
                from scipy.interpolate import CubicSpline

                spline = CubicSpline(y, x)
                sample_y = np.linspace(y[0], y[-1], max(2, int(round(y[-1] - y[0])) + 1))
                sample_x = spline(sample_y)
                valid = (sample_x >= 0) & (sample_x < width)
                smooth = np.stack([sample_x[valid], sample_y[valid]], axis=1).astype(np.int32)
                if len(smooth) >= 2:
                    cv2.polylines(mask, [smooth], False, lane_id, thickness=thickness)
                    continue
            except Exception:
                pass

        pts_i = pts.astype(np.int32)
        cv2.polylines(mask, [pts_i], False, lane_id, thickness=thickness)
    return mask


def make_sample(json_path, image_exts):
    data = read_labelme(json_path)
    image_name = data.get("imagePath") or (json_path.stem + ".jpg")
    image_path = json_path.parent / image_name
    if not image_path.exists():
        for ext in image_exts:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                image_name = candidate.name
                break
    if not image_path.exists():
        return None

    width = int(data.get("imageWidth", 0))
    height = int(data.get("imageHeight", 0))
    if width <= 0 or height <= 0:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        height, width = image.shape[:2]

    lanes = collect_lane_points(data)
    return {
        "json_path": json_path,
        "image_path": image_path,
        "image_name": image_path.name,
        "stem": image_path.stem,
        "width": width,
        "height": height,
        "lanes": lanes,
    }


def split_samples(samples, train_ratio, val_ratio, seed):
    groups = {}
    for sample in samples:
        prefix = sample["stem"].split("_")[0]
        groups.setdefault(prefix, []).append(sample)

    train, val, test = [], [], []
    rng = random.Random(seed)
    for prefix in sorted(groups):
        items = groups[prefix]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


def write_subset(name, items, out_dir, thickness):
    lines = []
    for sample in items:
        image_dst = out_dir / "images" / sample["image_name"]
        label_dst = out_dir / "laneseg_label_w16" / f"{sample['stem']}.png"
        annot_dst = out_dir / "annotations" / f"{sample['stem']}.lines.txt"

        shutil.copy2(sample["image_path"], image_dst)
        mask = draw_lane_mask(sample["height"], sample["width"], sample["lanes"], thickness)
        cv2.imwrite(str(label_dst), mask)

        with open(annot_dst, "w", encoding="utf-8") as f:
            for lane_id in sorted(sample["lanes"]):
                coords = " ".join(f"{x:.1f} {y:.1f}" for x, y in sample["lanes"][lane_id])
                f.write(coords + "\n")

        existence = ["1" if lane_id in sample["lanes"] else "0" for lane_id in range(1, 5)]
        lines.append(f"/images/{sample['image_name']} /laneseg_label_w16/{sample['stem']}.png {' '.join(existence)}")

    list_dir = out_dir / "list"
    with open(list_dir / f"{name}_gt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if name == "train":
        shutil.copy2(list_dir / "train_gt.txt", list_dir / "train.txt")
    elif name == "val":
        with open(list_dir / "val.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(line.split()[0] for line in lines))
    elif name == "test":
        with open(list_dir / "test.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(line.split()[0] for line in lines))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="LabelMe image/json directory")
    parser.add_argument("--out", required=True, help="Output CULane-style directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thickness", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ["images", "laneseg_label_w16", "annotations", "list"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    samples = []
    skipped = 0
    for json_path in sorted(root.rglob("*.json")):
        sample = make_sample(json_path, image_exts)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    report = []
    for name, items in splits.items():
        lines = write_subset(name, items, out_dir, args.thickness)
        lane_counts = {lane_id: 0 for lane_id in range(1, 5)}
        for sample in items:
            for lane_id in sample["lanes"]:
                lane_counts[lane_id] += 1
        report.append(f"{name}: {len(items)} samples, {len(lines)} lines, lanes={lane_counts}")

    with open(out_dir / "list" / "split_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report + [f"skipped: {skipped}"]))

    print("Converted LabelMe lanes to CULane style")
    print(f"Output: {out_dir}")
    for line in report:
        print(line)
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
