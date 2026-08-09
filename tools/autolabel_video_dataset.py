import argparse
import contextlib
import csv
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
UFLD_ROOT = PROJECT_ROOT / "third_party" / "ufld"
if str(UFLD_ROOT) not in sys.path:
    sys.path.insert(0, str(UFLD_ROOT))

from data.constant import culane_row_anchor
from model.model import parsingNet
from train_emergency_boundary_baseline import band_logits, build_model_from_config
from tools.boundary_polyline import mask_to_polyline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDE_DIR_PREFIXES = (
    "_autolabel_preview",
    "_backup_before_autolabel",
    "_backup_before_resize",
    "_raw_boundary_predictions",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label a video-image dataset into LabelMe JSON files.")
    parser.add_argument("--root", default="data_to_label")
    parser.add_argument("--lane-model", default="weights/project_lane_detector.pth")
    parser.add_argument("--boundary-config", default="configs/main/a1_band_edge03.yaml")
    parser.add_argument(
        "--boundary-checkpoint",
        default="weights/a1_band_edge03.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-boundary-points", type=int, default=5)
    parser.add_argument("--max-line-points", type=int, default=20)
    parser.add_argument("--min-lane-points", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--write-in-place",
        action="store_true",
        help="Write same-name JSON files beside source images without resizing or replacing the images.",
    )
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def is_excluded(path, root):
    rel_parts = path.relative_to(root).parts
    return any(part.startswith(EXCLUDE_DIR_PREFIXES) for part in rel_parts)


def find_images(root):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_excluded(p, root)
    )


def find_jsons(root):
    return sorted(
        p for p in root.rglob("*.json")
        if p.is_file() and not is_excluded(p, root)
    )


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    return state


def build_lane_model(model_path, device):
    net = parsingNet(
        pretrained=False,
        backbone="18",
        cls_dim=(201, 18, 4),
        use_aux=False,
    )
    net.load_state_dict(load_checkpoint(model_path), strict=False)
    net.to(device)
    net.eval()
    return net


def predict_lanes(net, image, device, min_points):
    transform = transforms.Compose(
        [
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img_w, img_h = image.size
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()):
            out = net(tensor)

    griding_num = 200
    cls_num_per_lane = 18
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].detach().cpu().numpy()[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1], axis=0)
    idx = np.arange(griding_num) + 1
    loc = np.sum(prob * idx.reshape(-1, 1, 1), axis=0)
    out_cls = np.argmax(out_j, axis=0)
    loc[out_cls == griding_num] = 0

    lanes = []
    for lane_idx in range(4):
        points = []
        for k in range(loc.shape[0]):
            if loc[k, lane_idx] <= 0:
                continue
            x = float(loc[k, lane_idx] * col_sample_w * img_w / 800.0 - 1.0)
            y = float(img_h * (culane_row_anchor[cls_num_per_lane - 1 - k] / 288.0) - 1.0)
            if 0 <= x < img_w and 0 <= y < img_h:
                points.append([round(x, 2), round(y, 2)])
        if len(points) >= min_points:
            lanes.append((f"lane{lane_idx + 1}", thin_points(points, 12)))
    return lanes


def build_boundary_model(config_path, checkpoint_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()
    mean = torch.tensor(cfg["dataset"]["image_mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg["dataset"]["image_std"], dtype=torch.float32).view(3, 1, 1)
    image_size = tuple(cfg["dataset"]["image_size"])
    return model, mean, std, image_size


def predict_boundary(model, image, mean, std, image_size, device, threshold, max_points, min_points):
    target_h, target_w = image_size
    img = image.convert("RGB")
    original_w, original_h = img.size
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        prob = torch.sigmoid(band_logits(outputs))[0, 0].detach().cpu().numpy()
    _, _, points, _ = mask_to_polyline(prob, threshold, max_points)
    if len(points) < min_points:
        return []
    scale_x = original_w / target_w
    scale_y = original_h / target_h
    return [[round(float(x) * scale_x, 2), round(float(y) * scale_y, 2)] for x, y in points]


def thin_points(points, max_points):
    if len(points) <= max_points:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in points]
    idx = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    selected = [points[i] for i in idx]
    deduped = []
    for x, y in selected:
        if deduped and abs(deduped[-1][1] - y) < 1e-6:
            continue
        deduped.append([round(float(x), 2), round(float(y), 2)])
    return deduped


def line_shape(label, points, auto=False):
    flags = {"auto_label": True} if auto else {}
    return {
        "label": label,
        "line_color": None,
        "fill_color": None,
        "points": points,
        "group_id": None,
        "shape_type": "linestrip",
        "flags": flags,
    }


def make_labelme(image_path, image_size, lanes, boundary_points):
    shapes = [line_shape(label, points, auto=True) for label, points in lanes]
    if boundary_points:
        shapes.append(line_shape("right_boundary", boundary_points, auto=True))
    w, h = image_size
    return {
        "version": "3.16.2",
        "flags": {"auto_label": True},
        "shapes": shapes,
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def normalize_existing_json(json_path, image_path, out_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as img:
        actual_w, actual_h = img.size
    old_w = int(data.get("imageWidth") or actual_w)
    old_h = int(data.get("imageHeight") or actual_h)
    scaled = False
    sx = actual_w / old_w if old_w > 0 else 1.0
    sy = actual_h / old_h if old_h > 0 else 1.0
    if old_w != actual_w or old_h != actual_h:
        for shape in data.get("shapes", []):
            new_points = []
            for x, y in shape.get("points", []):
                new_points.append([float(x) * sx, float(y) * sy])
            shape["points"] = new_points
        scaled = True
    for shape in data.get("shapes", []):
        shape.setdefault("line_color", None)
        shape.setdefault("fill_color", None)
        shape.setdefault("flags", {})
        shape.setdefault("shape_type", "linestrip")
    data["imagePath"] = image_path.name
    data["imageData"] = None
    data["imageWidth"] = actual_w
    data["imageHeight"] = actual_h
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return scaled, old_w, old_h, actual_w, actual_h


def copy_image_to_preview(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB")
        if image.size != (800, 450):
            image = image.resize((800, 450), Image.BILINEAR)
        image.save(dst, quality=95)


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / f"_backup_before_autolabel_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    preview_dir = None
    if not args.write_in_place:
        preview_dir = root / f"_autolabel_preview_{timestamp}"
        preview_dir.mkdir(parents=True, exist_ok=False)

    images = find_images(root)
    jsons = find_jsons(root)
    for json_path in jsons:
        rel = json_path.relative_to(root)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, dst)

    device = resolve_device(args.device)
    lane_model_path = resolve_path(args.lane_model)
    boundary_config = resolve_path(args.boundary_config)
    boundary_checkpoint = resolve_path(args.boundary_checkpoint)
    lane_net = build_lane_model(lane_model_path, device)
    boundary_model, boundary_mean, boundary_std, boundary_image_size = build_boundary_model(
        boundary_config, boundary_checkpoint, device
    )

    rows = []
    created = 0
    copied_existing = 0
    normalized_existing = 0
    no_boundary = 0
    errors = 0

    print(f"ROOT={root}")
    print(f"BACKUP={backup_dir}")
    print(f"MODE={'write_in_place' if args.write_in_place else 'preview'}")
    if preview_dir is not None:
        print(f"PREVIEW={preview_dir}")
    print(f"IMAGES={len(images)} EXISTING_JSON={len(jsons)} DEVICE={device}")
    print(f"LANE_MODEL={lane_model_path}")
    print(f"BOUNDARY_MODEL={boundary_checkpoint}")

    for idx, image_path in enumerate(images, 1):
        rel = image_path.relative_to(root)
        source_json = image_path.with_suffix(".json")
        output_image = image_path if args.write_in_place else preview_dir / rel
        output_json = source_json if args.write_in_place else output_image.with_suffix(".json")
        try:
            if not args.write_in_place:
                copy_image_to_preview(image_path, output_image)
            with Image.open(output_image) as image:
                image = image.convert("RGB")
                if source_json.exists():
                    if args.write_in_place:
                        copied_existing += 1
                        rows.append({
                            "image": str(rel),
                            "status": "existing_json_preserved",
                            "lanes": "",
                            "right_boundary": "",
                            "normalized": 0,
                            "old_size": f"{image.size[0]}x{image.size[1]}",
                            "new_size": f"{image.size[0]}x{image.size[1]}",
                            "error": "",
                        })
                        continue
                    scaled, old_w, old_h, actual_w, actual_h = normalize_existing_json(
                        source_json, output_image, output_json
                    )
                    copied_existing += 1
                    if scaled:
                        normalized_existing += 1
                    rows.append({
                        "image": str(rel),
                        "status": "existing_json_copied",
                        "lanes": "",
                        "right_boundary": "",
                        "normalized": int(scaled),
                        "old_size": f"{old_w}x{old_h}",
                        "new_size": f"{actual_w}x{actual_h}",
                        "error": "",
                    })
                else:
                    lanes = predict_lanes(lane_net, image, device, args.min_lane_points)
                    boundary = predict_boundary(
                        boundary_model,
                        image,
                        boundary_mean,
                        boundary_std,
                        boundary_image_size,
                        device,
                        args.threshold,
                        args.max_line_points,
                        args.min_boundary_points,
                    )
                    data = make_labelme(output_image, image.size, lanes, boundary)
                    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    created += 1
                    if not boundary:
                        no_boundary += 1
                    rows.append({
                        "image": str(rel),
                        "status": "auto_labeled",
                        "lanes": "|".join(label for label, _ in lanes),
                        "right_boundary": int(bool(boundary)),
                        "normalized": 0,
                        "old_size": "",
                        "new_size": f"{image.size[0]}x{image.size[1]}",
                        "error": "",
                    })
        except Exception as exc:
            errors += 1
            rows.append({
                "image": str(rel),
                "status": "error",
                "lanes": "",
                "right_boundary": "",
                "normalized": "",
                "old_size": "",
                "new_size": "",
                "error": str(exc),
            })
            print(f"ERROR {rel}: {exc}")

        if idx % 50 == 0 or idx == len(images):
            print(
                f"PROGRESS={idx}/{len(images)} CREATED={created} COPIED_EXISTING={copied_existing} "
                f"NORMALIZED_EXISTING={normalized_existing} NO_BOUNDARY={no_boundary} ERRORS={errors}"
            )

    report_root = preview_dir if preview_dir is not None else root
    report_suffix = "" if preview_dir is not None else f"_{timestamp}"
    report_csv = report_root / f"autolabel_summary{report_suffix}.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "status", "lanes", "right_boundary", "normalized", "old_size", "new_size", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_md = report_root / f"autolabel_report{report_suffix}.md"
    report_md.write_text(
        "\n".join(
            [
                "# Auto Label Report",
                "",
                f"- Source root: `{root}`",
                f"- Backup dir: `{backup_dir}`",
                f"- Mode: `{'write_in_place' if args.write_in_place else 'preview'}`",
                f"- Preview dir: `{preview_dir}`",
                f"- Total images: {len(images)}",
                f"- Existing JSON copied: {copied_existing}",
                f"- Existing JSON normalized in preview: {normalized_existing}",
                f"- New auto-labeled JSON: {created}",
                f"- Auto-labeled without right_boundary: {no_boundary}",
                f"- Errors: {errors}",
                f"- Lane model: `{lane_model_path}`",
                f"- Boundary config: `{boundary_config}`",
                f"- Boundary checkpoint: `{boundary_checkpoint}`",
                "",
                "Source images were not resized or overwritten.",
                "Existing JSON files were preserved; only missing same-name JSON files were generated.",
            ]
        ),
        encoding="utf-8",
    )

    print("DONE")
    print(f"BACKUP={backup_dir}")
    if preview_dir is not None:
        print(f"PREVIEW={preview_dir}")
    print(f"REPORT={report_md}")
    print(f"SUMMARY={report_csv}")


if __name__ == "__main__":
    main()
