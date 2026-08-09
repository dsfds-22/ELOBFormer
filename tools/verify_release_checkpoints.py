from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def state_dict(checkpoint: Any) -> dict[str, Any]:
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify release checkpoint tensors.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=Path("."))
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for record in records:
        source = torch.load(
            args.source_root / record["source"],
            map_location="cpu",
            weights_only=False,
        )
        release = torch.load(
            args.release_root / record["destination"],
            map_location="cpu",
            weights_only=False,
        )
        source_state = state_dict(source)
        release_state = state_dict(release)
        if source_state.keys() != release_state.keys():
            failures.append(f"{record['model_id']}: state-dict keys differ")
            continue
        for key in source_state:
            source_value = source_state[key]
            release_value = release_state[key]
            if torch.is_tensor(source_value):
                if not torch.equal(source_value, release_value):
                    failures.append(f"{record['model_id']}: tensor differs: {key}")
                    break
            elif source_value != release_value:
                failures.append(f"{record['model_id']}: value differs: {key}")
                break
        else:
            print(f"OK {record['model_id']} ({len(source_state)} entries)")

    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    main()
