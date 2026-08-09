import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
UFLD_ROOT = PROJECT_ROOT / "third_party" / "ufld"
if str(UFLD_ROOT) not in sys.path:
    sys.path.insert(0, str(UFLD_ROOT))

from data.constant import culane_row_anchor, tusimple_row_anchor
from model.model import parsingNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-label lane1-lane4 as LabelMe linestrip shapes."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing images.")
    parser.add_argument(
        "--model",
        default="weights/project_lane_detector.pth",
        help="Lane model checkpoint, e.g. culane_18.pth or a fine-tuned checkpoint.",
    )
    parser.add_argument("--dataset", default="CULane", choices=["CULane", "Tusimple"])
    parser.add_argument("--backbone", default="18")
    parser.add_argument("--griding-num", type=int, default=200)
    parser.add_argument("--num-lanes", type=int, default=4)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=0,
        help="Drop predicted points closer than this many pixels to image left/right edge.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files. By default existing JSON files are skipped.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def build_model(args, device):
    cls_num_per_lane = 18 if args.dataset == "CULane" else 56
    net = parsingNet(
        pretrained=False,
        backbone=args.backbone,
        cls_dim=(args.griding_num + 1, cls_num_per_lane, args.num_lanes),
        use_aux=False,
    )
    net.load_state_dict(load_checkpoint(args.model), strict=False)
    net.to(device)
    net.eval()
    return net, cls_num_per_lane


def predict_lanes(net, image, args, cls_num_per_lane, row_anchor, device):
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
        # model/model.py prints debug tensor shapes; keep batch labeling readable.
        with contextlib.redirect_stdout(io.StringIO()):
            out = net(tensor)

    col_sample = np.linspace(0, 800 - 1, args.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].detach().cpu().numpy()[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1], axis=0)
    idx = np.arange(args.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == args.griding_num] = 0

    lanes = []
    for lane_idx in range(args.num_lanes):
        points = []
        for k in range(loc.shape[0]):
            if loc[k, lane_idx] == 0:
                continue
            x = float(loc[k, lane_idx] * col_sample_w * img_w / 800.0 - 1.0)
            y = float(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288.0) - 1.0)
            if x < args.edge_margin or x > img_w - 1 - args.edge_margin:
                continue
            if y < 0 or y > img_h - 1:
                continue
            points.append([round(x, 2), round(y, 2)])
        if len(points) >= args.min_points:
            lanes.append((f"lane{lane_idx + 1}", points))
    return lanes


def make_labelme_json(image_path, image_size, lanes):
    img_w, img_h = image_size
    return {
        "version": "3.16.2",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": points,
                "shape_type": "linestrip",
                "flags": {},
            }
            for label, points in lanes
        ],
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    device = resolve_device(args.device)
    row_anchor = culane_row_anchor if args.dataset == "CULane" else tusimple_row_anchor
    net, cls_num_per_lane = build_model(args, device)

    image_files = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    created = 0
    skipped_existing = 0
    no_lane = 0
    errors = 0

    print(f"IMAGE_DIR={image_dir}")
    print(f"MODEL={Path(args.model).resolve()}")
    print(f"DEVICE={device}")
    print(f"IMAGES={len(image_files)}")

    for idx, image_path in enumerate(image_files, 1):
        json_path = image_path.with_suffix(".json")
        if json_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                lanes = predict_lanes(net, image, args, cls_num_per_lane, row_anchor, device)
                data = make_labelme_json(image_path, image.size, lanes)

            if not lanes:
                no_lane += 1

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            created += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR {image_path.name}: {exc}")

        if idx % 50 == 0 or idx == len(image_files):
            print(
                f"PROGRESS={idx}/{len(image_files)} CREATED={created} "
                f"SKIPPED_EXISTING={skipped_existing} NO_LANE={no_lane} ERRORS={errors}"
            )

    print("DONE")
    print(f"CREATED_OR_OVERWRITTEN={created}")
    print(f"SKIPPED_EXISTING={skipped_existing}")
    print(f"NO_LANE_JSONS={no_lane}")
    print(f"ERRORS={errors}")


if __name__ == "__main__":
    main()
