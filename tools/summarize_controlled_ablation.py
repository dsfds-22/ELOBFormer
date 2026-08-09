import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


GROUPS = ("C0", "C1", "C2")
SEEDS = (42, 2026, 3407)
SUMMARY_METRICS = (
    "all_iou",
    "all_dice",
    "positive_iou",
    "positive_dice",
    "boundary_f1_positive",
    "pmbe_positive",
    "cmbe_positive",
    "positive_zero_prediction_rate",
    "negative_false_positive_rate",
    "row_coverage_positive",
    "continuity_positive",
)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_std(values):
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def group_name(model):
    return model.split("_", 1)[0]


def seed_value(model):
    return int(model.rsplit("seed", 1)[1])


def validation_summary(run_root):
    rows = []
    for group in GROUPS:
        for seed in SEEDS:
            metrics = read_csv(run_root / f"{group.lower()}_seed{seed}" / "metrics.csv")
            best = max(metrics, key=lambda row: float(row["val_iou"]))
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "epochs_run": len(metrics),
                    "best_epoch": int(best["epoch"]),
                    "best_val_iou": float(best["val_iou"]),
                    "best_val_dice": float(best["val_dice"]),
                }
            )
    return rows


def aggregate_test(overall_rows):
    grouped = defaultdict(list)
    for row in overall_rows:
        grouped[group_name(row["model"])].append(row)
    output = []
    for group in GROUPS:
        items = grouped[group]
        result = {"group": group, "seeds": len(items)}
        for metric in SUMMARY_METRICS:
            values = [float(item[metric]) for item in items]
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_sd"] = sample_std(values)
        output.append(result)
    return output


def paired_seed_deltas(overall_rows, left, right):
    lookup = {row["model"]: row for row in overall_rows}
    output = []
    for seed in SEEDS:
        left_row = lookup[f"{left}_seed{seed}"]
        right_row = lookup[f"{right}_seed{seed}"]
        output.append(
            {
                "comparison": f"{right}-{left}",
                "seed": seed,
                "delta_bf1": float(right_row["boundary_f1_positive"])
                - float(left_row["boundary_f1_positive"]),
                "delta_positive_iou": float(right_row["positive_iou"])
                - float(left_row["positive_iou"]),
                "delta_pmbe": float(right_row["pmbe_positive"])
                - float(left_row["pmbe_positive"]),
                "delta_cmbe": float(right_row["cmbe_positive"])
                - float(left_row["cmbe_positive"]),
                "delta_empty_rate": float(right_row["positive_zero_prediction_rate"])
                - float(left_row["positive_zero_prediction_rate"]),
                "delta_negative_fpr": float(right_row["negative_false_positive_rate"])
                - float(left_row["negative_false_positive_rate"]),
            }
        )
    return output


def cluster_bootstrap(per_image_rows, left, right, repetitions, rng):
    positives = [row for row in per_image_rows if int(row["has_boundary"]) == 1]
    lookup = {
        (row["model"], row["sample_id"]): row
        for row in positives
    }
    samples_by_scene = defaultdict(list)
    reference_model = f"{left}_seed{SEEDS[0]}"
    for row in positives:
        if row["model"] == reference_model:
            samples_by_scene[row["scene_group"]].append(row["sample_id"])
    scenes = sorted(samples_by_scene)
    if not scenes:
        raise RuntimeError("No positive scene groups found")

    metric_columns = {
        "delta_bf1": "boundary_f1",
        "delta_pmbe": "penalized_mean_boundary_error",
        "delta_iou": "iou",
    }
    observed = {}
    values_by_scene = {}
    for output_name, column in metric_columns.items():
        scene_values = {}
        all_deltas = []
        for scene, sample_ids in samples_by_scene.items():
            deltas = []
            for seed in SEEDS:
                for sample_id in sample_ids:
                    lrow = lookup[(f"{left}_seed{seed}", sample_id)]
                    rrow = lookup[(f"{right}_seed{seed}", sample_id)]
                    deltas.append(float(rrow[column]) - float(lrow[column]))
            scene_values[scene] = np.asarray(deltas, dtype=np.float64)
            all_deltas.extend(deltas)
        observed[output_name] = float(np.mean(all_deltas))
        values_by_scene[output_name] = scene_values

    boot = {name: np.empty(repetitions, dtype=np.float64) for name in metric_columns}
    for index in range(repetitions):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        for name in metric_columns:
            concatenated = np.concatenate([values_by_scene[name][scene] for scene in selected])
            boot[name][index] = float(np.mean(concatenated))

    result = {
        "comparison": f"{right}-{left}",
        "cluster_field": "scene_group",
        "clusters": len(scenes),
        "repetitions": repetitions,
    }
    for name in metric_columns:
        result[f"{name}_observed"] = observed[name]
        result[f"{name}_ci_low"] = float(np.percentile(boot[name], 2.5))
        result[f"{name}_ci_high"] = float(np.percentile(boot[name], 97.5))
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    evaluation_dir = Path(args.evaluation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = read_csv(evaluation_dir / "overall_metrics.csv")
    per_image_rows = read_csv(evaluation_dir / "per_image_metrics.csv")
    val_rows = validation_summary(run_root)
    test_summary = aggregate_test(overall_rows)

    comparisons = (("C0", "C1"), ("C1", "C2"), ("C0", "C2"))
    paired_rows = []
    for left, right in comparisons:
        paired_rows.extend(paired_seed_deltas(overall_rows, left, right))

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows = [
        cluster_bootstrap(
            per_image_rows,
            left,
            right,
            args.bootstrap_repetitions,
            rng,
        )
        for left, right in comparisons
    ]

    write_csv(out_dir / "validation_runs.csv", val_rows)
    write_csv(out_dir / "test_seed_summary.csv", test_summary)
    write_csv(out_dir / "paired_seed_deltas.csv", paired_rows)
    write_csv(out_dir / "scene_cluster_bootstrap.csv", bootstrap_rows)

    payload = {
        "validation_runs": val_rows,
        "test_seed_summary": test_summary,
        "paired_seed_deltas": paired_rows,
        "scene_cluster_bootstrap": bootstrap_rows,
    }
    (out_dir / "controlled_ablation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
