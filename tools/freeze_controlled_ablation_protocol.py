"""Freeze all training inputs for the 2026-08-07 controlled ablation."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "controlled_ablation_20260807"
DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "emergency_outer_boundary_grouped_fixed12_v2_clean601"
)
CONFIGS = {
    "C0": PROJECT_ROOT / "configs" / "controlled_ablation_c0_shallow_band.yaml",
    "C1": PROJECT_ROOT / "configs" / "controlled_ablation_c1_deep_band_edge0.yaml",
    "C2": PROJECT_ROOT / "configs" / "controlled_ablation_c2_deep_band_edge03.yaml",
}
SEEDS = (42, 2026, 3407)
STAGES = ((1, 27), (2, 54), (3, 80))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the existing manifests instead of creating them.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def build_run_matrix():
    rows = []
    for group, config in CONFIGS.items():
        for seed in SEEDS:
            output = (
                PROJECT_ROOT
                / "runs"
                / "emergency_boundary"
                / "controlled_ablation_20260807"
                / f"{group.lower()}_seed{seed}"
            )
            if output.exists():
                raise FileExistsError(
                    f"Registered output directory must not exist before training: {output}"
                )
            for stage, target_epoch in STAGES:
                resume = ""
                if stage > 1:
                    resume = (
                        f' --resume "{relative(output / "checkpoints" / "last.pt")}"'
                    )
                command = (
                    f'$env:PYTHONHASHSEED="{seed}"; python '
                    f'train_emergency_boundary_baseline.py --config "{relative(config)}" '
                    f'--seed {seed} --epochs {target_epoch} '
                    f'--output-dir "{relative(output)}" --skip-final-test{resume}'
                )
                rows.append(
                    {
                        "group": group,
                        "seed": seed,
                        "stage": stage,
                        "target_epoch": target_epoch,
                        "config": relative(config),
                        "output_dir": relative(output),
                        "resume_checkpoint": (
                            ""
                            if stage == 1
                            else relative(output / "checkpoints" / "last.pt")
                        ),
                        "status": "pending",
                        "command": command,
                    }
                )
    return rows


def build_dataset_manifest():
    records = []
    split_counts = {}
    for split in ("train", "val", "test"):
        csv_path = DATASET_ROOT / "splits" / f"{split}.csv"
        rows = read_csv(csv_path)
        split_counts[split] = len(rows)
        for row in rows:
            for role, field in (
                ("image", "image"),
                ("labelme_json", "json"),
                ("band_mask", "band_mask"),
                ("edge_mask", "edge_mask"),
                ("heatmap", "heatmap"),
            ):
                raw = str(row.get(field, "")).strip()
                if not raw:
                    continue
                path = Path(raw)
                if not path.is_absolute():
                    path = DATASET_ROOT / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                digest = sha256(path)
                expected_field = {
                    "image": "image_sha256",
                    "labelme_json": "json_sha256",
                }.get(role)
                expected = str(row.get(expected_field, "")).strip() if expected_field else ""
                if expected and expected.lower() != digest.lower():
                    raise AssertionError(
                        f"Stored hash mismatch: {split}/{row['sample_id']} {role}"
                    )
                records.append(
                    {
                        "split": split,
                        "sample_id": row["sample_id"],
                        "scene_segment": row["scene_segment"],
                        "role": role,
                        "path": relative(path),
                        "bytes": path.stat().st_size,
                        "sha256": digest,
                    }
                )
    return records, split_counts


def static_artifacts(run_matrix_path):
    paths = [
        PROJECT_ROOT / "train_emergency_boundary_baseline.py",
        PROJECT_ROOT / "tools" / "preflight_controlled_ablation.py",
        PROJECT_ROOT / "tools" / "verify_deterministic_resume.py",
        PROJECT_ROOT / "tools" / "freeze_controlled_ablation_protocol.py",
        REPORT_ROOT / "preregistration.yaml",
        REPORT_ROOT / "amendment_001_segmented_resume.yaml",
        REPORT_ROOT / "initialization_preflight.json",
        REPORT_ROOT / "resume_equivalence_check.json",
        run_matrix_path,
        PROJECT_ROOT / "segformer-B1.pth",
        DATASET_ROOT / "conversion_summary.json",
    ]
    paths.extend(CONFIGS.values())
    paths.extend(DATASET_ROOT / "splits" / f"{split}.csv" for split in ("train", "val", "test"))
    return paths


def create_manifests():
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads(
        (REPORT_ROOT / "initialization_preflight.json").read_text(encoding="utf-8")
    )
    if preflight.get("status") != "passed":
        raise RuntimeError("Initialization preflight has not passed.")
    registration = yaml.safe_load(
        (REPORT_ROOT / "preregistration.yaml").read_text(encoding="utf-8")
    )
    if registration["experiment"]["status"] != "frozen_before_training":
        raise RuntimeError("Preregistration status is not frozen_before_training.")

    run_matrix_path = REPORT_ROOT / "run_matrix.csv"
    run_rows = build_run_matrix()
    write_csv(
        run_matrix_path,
        run_rows,
        [
            "group",
            "seed",
            "stage",
            "target_epoch",
            "config",
            "output_dir",
            "resume_checkpoint",
            "status",
            "command",
        ],
    )

    dataset_records, split_counts = build_dataset_manifest()
    dataset_manifest = REPORT_ROOT / "dataset_referenced_files_sha256.csv"
    write_csv(
        dataset_manifest,
        dataset_records,
        ["split", "sample_id", "scene_segment", "role", "path", "bytes", "sha256"],
    )

    artifact_rows = []
    for path in static_artifacts(run_matrix_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_rows.append(
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    artifact_manifest = REPORT_ROOT / "frozen_artifacts_sha256.csv"
    write_csv(artifact_manifest, artifact_rows, ["path", "bytes", "sha256"])

    summary = {
        "status": "frozen_before_training",
        "experiment": "controlled_ablation_20260807",
        "python_executable": sys.executable,
        "split_counts": split_counts,
        "dataset_manifest_records": len(dataset_records),
        "dataset_manifest_sha256": sha256(dataset_manifest),
        "artifact_manifest_records": len(artifact_rows),
        "artifact_manifest_sha256": sha256(artifact_manifest),
        "run_count": len(CONFIGS) * len(SEEDS),
        "segmented_command_count": len(run_rows),
        "stage_target_epochs": [target for _, target in STAGES],
        "seeds": list(SEEDS),
        "cluster_field": "scene_segment",
        "final_test_allowed": False,
    }
    (REPORT_ROOT / "freeze_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def verify_manifests():
    artifact_manifest = REPORT_ROOT / "frozen_artifacts_sha256.csv"
    dataset_manifest = REPORT_ROOT / "dataset_referenced_files_sha256.csv"
    failures = []
    for manifest in (artifact_manifest, dataset_manifest):
        for row in read_csv(manifest):
            path = PROJECT_ROOT / row["path"]
            if not path.is_file():
                failures.append(f"missing:{row['path']}")
            elif sha256(path) != row["sha256"]:
                failures.append(f"changed:{row['path']}")
    if failures:
        raise AssertionError("\n".join(failures[:50]))
    print("CONTROLLED_ABLATION_FREEZE_VERIFY=PASSED")


def main():
    args = parse_args()
    if args.verify:
        verify_manifests()
    else:
        summary = create_manifests()
        print("CONTROLLED_ABLATION_FREEZE=CREATED")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
