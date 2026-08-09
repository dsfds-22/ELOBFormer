"""Preflight the registered C0/C1/C2 controlled ablation experiment."""

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_emergency_boundary_baseline as training  # noqa: E402


DEFAULT_CONFIGS = {
    "C0": PROJECT_ROOT / "configs" / "controlled_ablation_c0_shallow_band.yaml",
    "C1": PROJECT_ROOT / "configs" / "controlled_ablation_c1_deep_band_edge0.yaml",
    "C2": PROJECT_ROOT / "configs" / "controlled_ablation_c2_deep_band_edge03.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="reports/controlled_ablation_20260807",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def canonical_without_allowed_differences(config):
    value = copy.deepcopy(config)
    value["model"]["name"] = "<allowed-model-name>"
    value["loss"]["edge"] = "<allowed-edge-weight>"
    value["train"]["output_dir"] = "<allowed-output-dir>"
    return value


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(state, prefixes=None):
    digest = hashlib.sha256()
    keys = sorted(state)
    if prefixes is not None:
        keys = [key for key in keys if any(key.startswith(prefix) for prefix in prefixes)]
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(tensor_hash(state[key]).encode("ascii"))
    return digest.hexdigest()


def load_initialized_model(config, seed):
    training.seed_everything(seed, deterministic=True)
    model = training.build_model_from_config(config)
    checkpoint = PROJECT_ROOT / config["model"]["pretrained_backbone"]
    load_info = training.load_segformer_pretrained_backbone(model, checkpoint)
    return model, load_info


def read_split_rows(dataset_root, relative_csv):
    path = dataset_root / relative_csv
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def dataset_audit(config):
    dataset = config["dataset"]
    root = PROJECT_ROOT / dataset["root"]
    split_rows = {
        split: read_split_rows(root, dataset[f"{split}_csv"])
        for split in ("train", "val", "test")
    }
    expected = {"train": 3002, "val": 524, "test": 601}
    counts = {split: len(rows) for split, rows in split_rows.items()}
    if counts != expected:
        raise AssertionError(f"Split counts differ: {counts} != {expected}")

    scene_sets = {
        split: {row["scene_segment"] for row in rows if row.get("scene_segment")}
        for split, rows in split_rows.items()
    }
    overlaps = {
        "train_val": sorted(scene_sets["train"] & scene_sets["val"]),
        "train_test": sorted(scene_sets["train"] & scene_sets["test"]),
        "val_test": sorted(scene_sets["val"] & scene_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"scene_segment leakage detected: {overlaps}")
    return {
        "counts": counts,
        "positive_counts": {
            split: sum(int(row["has_boundary"]) for row in rows)
            for split, rows in split_rows.items()
        },
        "negative_counts": {
            split: sum(1 - int(row["has_boundary"]) for row in rows)
            for split, rows in split_rows.items()
        },
        "scene_segment_counts": {
            split: len(values) for split, values in scene_sets.items()
        },
        "scene_segment_overlaps": overlaps,
        "bootstrap_cluster_field": "scene_segment",
    }


def real_batch(config, count=4):
    dataset_cfg = config["dataset"]
    dataset = training.BoundaryDataset(
        PROJECT_ROOT / dataset_cfg["root"],
        dataset_cfg["train_csv"],
        dataset_cfg["image_size"],
        dataset_cfg["image_mean"],
        dataset_cfg["image_std"],
        train=False,
        augmentation="none",
    )
    positives = [i for i, row in enumerate(dataset.rows) if int(row["has_boundary"]) == 1]
    negatives = [i for i, row in enumerate(dataset.rows) if int(row["has_boundary"]) == 0]
    indices = positives[: count // 2] + negatives[: count - count // 2]
    return default_collate([dataset[index] for index in indices]), indices


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def grad_norm(model, prefixes):
    total = 0.0
    present = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes) and parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
            present += 1
    return total ** 0.5, present


def backward_check(model, batch, loss_weights, seed, device):
    model.train()
    model.zero_grad(set_to_none=True)
    training.seed_everything(seed, deterministic=True)
    amp_enabled = device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=amp_enabled,
    ):
        outputs = model(batch["image"])
        total, parts = training.compute_loss(
            outputs,
            batch,
            batch["valid"],
            loss_weights,
        )
    total.backward()
    shared_norm, shared_count = grad_norm(model, ("backbone.", "proj.", "fuse."))
    band_norm, band_count = grad_norm(model, ("band_head.",))
    edge_norm, edge_count = grad_norm(model, ("edge_head.",))
    return {
        "total_loss": float(total.detach().float().cpu().item()),
        "band_loss": float(parts["band_loss"].float().cpu().item()),
        "edge_loss_unweighted": float(parts["edge_loss"].float().cpu().item()),
        "shared_gradient_norm": shared_norm,
        "shared_gradient_tensor_count": shared_count,
        "band_head_gradient_norm": band_norm,
        "band_head_gradient_tensor_count": band_count,
        "edge_head_gradient_norm": edge_norm,
        "edge_head_gradient_tensor_count": edge_count,
    }


def environment(device):
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = {name: read_yaml(path) for name, path in DEFAULT_CONFIGS.items()}

    canonical = {
        name: canonical_without_allowed_differences(config)
        for name, config in configs.items()
    }
    if canonical["C0"] != canonical["C1"] or canonical["C1"] != canonical["C2"]:
        raise AssertionError("Controlled configs differ outside allowed fields.")
    if float(configs["C1"]["loss"]["edge"]) != 0.0:
        raise AssertionError("C1 edge weight is not 0.0")
    if float(configs["C2"]["loss"]["edge"]) != 0.3:
        raise AssertionError("C2 edge weight is not 0.3")
    if configs["C1"]["model"]["name"] != configs["C2"]["model"]["name"]:
        raise AssertionError("C1 and C2 model names differ.")

    audit = dataset_audit(configs["C1"])
    selected_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if selected_device == "auto":
        selected_device = "cpu"
    device = torch.device(selected_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    c1, c1_load = load_initialized_model(configs["C1"], args.seed)
    c2, c2_load = load_initialized_model(configs["C2"], args.seed)
    c1_state = c1.state_dict()
    c2_state = c2.state_dict()
    sections = {
        "full": None,
        "backbone": ("backbone.",),
        "decoder_projection": ("proj.",),
        "decoder_fusion": ("fuse.",),
        "band_head": ("band_head.",),
        "edge_head": ("edge_head.",),
    }
    hashes = {}
    for section, prefixes in sections.items():
        c1_hash = state_hash(c1_state, prefixes)
        c2_hash = state_hash(c2_state, prefixes)
        hashes[section] = {
            "c1": c1_hash,
            "c2": c2_hash,
            "identical": c1_hash == c2_hash,
        }
    if not all(item["identical"] for item in hashes.values()):
        raise AssertionError(f"C1/C2 initialization mismatch: {hashes}")

    raw_batch, selected_indices = real_batch(configs["C1"])
    batch = move_batch(raw_batch, device)
    c1 = c1.to(device)
    c2 = c2.to(device)
    c1.eval()
    c2.eval()
    with torch.no_grad():
        c1_band = c1(batch["image"])["band"]
        c2_band = c2(batch["image"])["band"]
    max_band_difference = float(
        (c1_band.float() - c2_band.float()).abs().max().cpu().item()
    )
    if max_band_difference != 0.0:
        raise AssertionError(f"Initial C1/C2 Band output differs: {max_band_difference}")

    c1_gradient = backward_check(
        c1, batch, configs["C1"]["loss"], args.seed, device
    )
    c2_gradient = backward_check(
        c2, batch, configs["C2"]["loss"], args.seed, device
    )
    if c1_gradient["edge_head_gradient_norm"] > 1e-12:
        raise AssertionError(f"C1 Edge gradient is non-zero: {c1_gradient}")
    if c2_gradient["edge_head_gradient_norm"] <= 0.0:
        raise AssertionError(f"C2 Edge gradient is zero: {c2_gradient}")

    result = {
        "status": "passed",
        "experiment": "controlled_ablation_20260807",
        "seed": args.seed,
        "device": str(device),
        "allowed_config_differences": ["model.name", "loss.edge", "train.output_dir"],
        "dataset": audit,
        "c1_edge_weight": float(configs["C1"]["loss"]["edge"]),
        "c2_edge_weight": float(configs["C2"]["loss"]["edge"]),
        "pretrained_loading": {
            "c1_loaded": c1_load["loaded"],
            "c1_skipped": c1_load["skipped"],
            "c2_loaded": c2_load["loaded"],
            "c2_skipped": c2_load["skipped"],
        },
        "initialization_hashes": hashes,
        "sample_indices": selected_indices,
        "sample_ids": list(raw_batch["id"]),
        "initial_band_max_abs_difference": max_band_difference,
        "c1_single_batch_gradient_check": c1_gradient,
        "c2_single_batch_gradient_check": c2_gradient,
        "environment": environment(device),
    }
    output_path = out_dir / "initialization_preflight.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("CONTROLLED_ABLATION_PREFLIGHT=PASSED")
    print(f"OUTPUT={output_path}")
    print(f"C1_C2_INITIAL_BAND_MAX_ABS_DIFF={max_band_difference:.12g}")
    print(f"C1_EDGE_GRAD_NORM={c1_gradient['edge_head_gradient_norm']:.12g}")
    print(f"C2_EDGE_GRAD_NORM={c2_gradient['edge_head_gradient_norm']:.12g}")


if __name__ == "__main__":
    main()
