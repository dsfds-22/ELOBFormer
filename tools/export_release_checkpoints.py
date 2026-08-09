from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint)!r}")
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise KeyError("Checkpoint has neither 'model' nor 'state_dict'")


def export_native(source: Path, destination: Path, model_id: str) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    payload = {
        "format_version": 1,
        "model_id": model_id,
        "model": model_state(checkpoint),
    }
    if isinstance(checkpoint, dict):
        for key in ("epoch", "metrics", "model_config", "presence_threshold", "threshold_mode", "val_metrics", "boundary_config", "image_size", "seed"):
            if key in checkpoint:
                payload[key] = checkpoint[key]
    if model_id == "v4_presence_verifier":
        payload["boundary_model_id"] = "a2_band_edge03_heatmap01"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return payload


def export_mmseg(source: Path, destination: Path, model_id: str) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    payload = {
        "meta": {"dataset_meta": source_meta.get("dataset_meta", {})},
        "state_dict": model_state(checkpoint),
        "model_id": model_id,
        "format_version": 1,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export optimizer-free release checkpoints.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mmseg", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    exported = []
    for record in records:
        source = Path(record["source"])
        destination = args.output_root / record["destination"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if record.get("format") == "mmseg" or args.mmseg:
            payload = export_mmseg(source, destination, record["model_id"])
        else:
            payload = export_native(source, destination, record["model_id"])
        exported.append(
            {
                **record,
                "destination": destination.as_posix(),
                "source_size": source.stat().st_size,
                "release_size": destination.stat().st_size,
                "source_sha256": sha256(source),
                "release_sha256": sha256(destination),
                "release_keys": sorted(payload),
            }
        )

    output = args.output_root / "checkpoint_manifest.json"
    output.write_text(json.dumps(exported, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
