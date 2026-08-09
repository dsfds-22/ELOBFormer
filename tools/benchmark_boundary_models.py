import argparse
import csv
import gc
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub
from train_emergency_boundary_baseline import (
    SegFormerBandOnlyDeployment,
    build_model_from_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMSEG_ROOT = PROJECT_ROOT / "mmsegmentation-main"


def read_ids(dataset_root, split, limit):
    csv_path = Path(dataset_root) / "splits" / f"{split}.csv"
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def load_images(rows, dataset_root, size_hw):
    root = Path(dataset_root)
    h, w = size_hw
    images = []
    for row in rows:
        image_path = root / row["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        images.append(image)
    return images


def tensor_from_image(image, mean, std, device):
    x = (
        torch.from_numpy(np.ascontiguousarray(image).copy())
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x.unsqueeze(0).to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_forward(
    name, model, input_tensor, iterations, forward, warmup, device
):
    model.eval()
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for i in range(warmup):
            _ = forward(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = forward(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_image_ms = elapsed * 1000.0 / iterations
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "model": name,
        "params_m": count_params(model) / 1e6,
        "images": iterations,
        "per_image_ms": per_image_ms,
        "fps": 1000.0 / per_image_ms if per_image_ms > 0 else 0.0,
        "peak_memory_mb": peak_memory_mb,
    }


def benchmark_torch_model(
    name,
    config_path,
    checkpoint_path,
    images,
    warmup,
    device,
    deploy_band_only=False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg).to(device)
    ckpt = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model"])

    mean = cfg["dataset"]["image_mean"]
    std = cfg["dataset"]["image_std"]
    input_tensor = tensor_from_image(images[0], mean, std, device)
    band_max_abs_diff = 0.0
    if deploy_band_only:
        full_model = model
        model = SegFormerBandOnlyDeployment(full_model).to(device).eval()
        full_model.eval()
        with torch.no_grad():
            full_outputs = full_model(input_tensor)
            full_band = (
                full_outputs["band"]
                if isinstance(full_outputs, dict)
                else full_outputs
            )
            deploy_band = model(input_tensor)
            band_max_abs_diff = float(
                (full_band - deploy_band).abs().max().item()
            )
        del full_outputs, full_band, deploy_band, full_model
    result = measure_forward(
        name,
        model,
        input_tensor,
        len(images),
        lambda x: model(x),
        warmup,
        device,
    )
    result["band_max_abs_diff"] = band_max_abs_diff
    del model, input_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def benchmark_mmseg_model(
    name, config_path, checkpoint_path, images, warmup, device
):
    os.environ["LANE_DETECTION_PROJECT_ROOT"] = str(PROJECT_ROOT)
    sys.path.insert(0, str(MMSEG_ROOT))

    from mmcv_lite_compat import install_mmcv_lite_ext_stub
    from run_mmseg_boundary import install_trusted_checkpoint_compat

    install_mmcv_lite_ext_stub()
    install_trusted_checkpoint_compat()

    from mmengine import Config
    from mmseg.apis import init_model

    config = Config.fromfile(str(Path(config_path).resolve()))
    model = init_model(
        config,
        str(Path(checkpoint_path).resolve()),
        device=str(device),
    )
    input_tensor = tensor_from_image(
        images[0],
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        device,
    )
    preprocessor_cfg = config.model.get(
        "data_preprocessor", config.get("data_preprocessor", {})
    )
    internal_size = tuple(
        int(value)
        for value in preprocessor_cfg.get(
            "size", input_tensor.shape[-2:]
        )
    )
    pad_height = internal_size[0] - input_tensor.shape[-2]
    pad_width = internal_size[1] - input_tensor.shape[-1]
    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            f"Internal MMSeg size {internal_size} is smaller than "
            f"input {tuple(input_tensor.shape[-2:])}."
        )
    if pad_height or pad_width:
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, pad_width, 0, pad_height), value=0.0
        )
    metadata = [
        {
            "img_shape": internal_size,
            "ori_shape": (450, 800),
            "pad_shape": internal_size,
        }
    ]
    result = measure_forward(
        name,
        model,
        input_tensor,
        len(images),
        lambda x: model.encode_decode(x, metadata),
        warmup,
        device,
    )
    del model, input_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="SegFormer-B1-current")
    parser.add_argument("--config", default="configs/main/a0_band_only.yaml")
    parser.add_argument(
        "--checkpoint",
        default="weights/a0_band_only.pt",
    )
    parser.add_argument("--out", default="outputs/model_runtime_benchmark.csv")
    parser.add_argument("--mmseg-name", default="")
    parser.add_argument("--mmseg-config", default="")
    parser.add_argument("--mmseg-checkpoint", default="")
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Benchmark only the requested MMSeg model.",
    )
    parser.add_argument(
        "--deploy-band-only",
        action="store_true",
        help="Skip training-only edge/heatmap heads and verify band equivalence.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    rows = read_ids(args.dataset_root, args.split, args.limit)
    images = load_images(rows, args.dataset_root, (450, 800))

    results = []
    if not args.skip_torch:
        results.append(
            benchmark_torch_model(
                args.name,
                args.config,
                args.checkpoint,
                images,
                args.warmup,
                device,
                deploy_band_only=args.deploy_band_only,
            )
        )
    if args.mmseg_config and args.mmseg_checkpoint:
        results.append(
            benchmark_mmseg_model(
                args.mmseg_name or "MMSeg-current",
                args.mmseg_config,
                args.mmseg_checkpoint,
                images,
                args.warmup,
                device,
            )
        )
    if not results:
        raise ValueError(
            "No model selected. Remove --skip-torch or provide an MMSeg model."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "params_m",
                "images",
                "per_image_ms",
                "fps",
                "peak_memory_mb",
                "band_max_abs_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    for result in results:
        print(
            f"{result['model']}: params={result['params_m']:.2f}M, "
            f"{result['per_image_ms']:.2f} ms/image, FPS={result['fps']:.2f}"
            f", peak_memory={result['peak_memory_mb']:.1f} MB"
            f", band_max_abs_diff={result.get('band_max_abs_diff', 0.0):.8f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
