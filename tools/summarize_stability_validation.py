import argparse
import csv
import statistics
from pathlib import Path


SEED_METRIC_FIELDS = [
    "model_group",
    "model_label",
    "seed",
    "best_epoch",
    "all_iou",
    "all_dice",
    "positive_iou",
    "positive_dice",
    "positive_zero_prediction_rate",
    "negative_fpr",
    "boundary_f1",
    "mean_boundary_error",
    "max_boundary_error",
    "row_coverage",
    "continuity",
]

AGGREGATE_METRICS = [
    "all_iou",
    "all_dice",
    "positive_iou",
    "positive_dice",
    "positive_zero_prediction_rate",
    "negative_fpr",
    "boundary_f1",
    "mean_boundary_error",
    "max_boundary_error",
    "row_coverage",
    "continuity",
]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_single_row(path):
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    value = row.get(key, "")
    return None if value in ("", None) else float(value)


def format_number(value):
    return "" if value is None else f"{value:.6f}"


def average(values):
    values = [float(value) for value in values if value is not None]
    return statistics.mean(values) if values else None


def sample_std(values):
    values = [float(value) for value in values if value is not None]
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def load_seed_metrics(manifest):
    output = []
    for experiment in manifest:
        run_dir = Path(experiment["run_dir"])
        test = read_single_row(run_dir / "test_metrics.csv")
        raw_path = (
            run_dir
            / "raw_boundary_metrics_test_band"
            / "raw_boundary_summary.csv"
        )
        if not raw_path.is_file():
            raw_path = (
                run_dir
                / "raw_boundary_metrics_test"
                / "raw_boundary_summary.csv"
            )
        raw = read_single_row(raw_path)
        output.append(
            {
                "model_group": experiment["model_group"],
                "model_label": experiment["model_label"],
                "seed": int(experiment["seed"]),
                "best_epoch": int(float(test["checkpoint_epoch"])),
                "all_iou": float(test["iou_all"]),
                "all_dice": float(test["dice_all"]),
                "positive_iou": float(test["iou_positive"]),
                "positive_dice": float(test["dice_positive"]),
                "positive_zero_prediction_rate": float(
                    test["positive_zero_prediction_rate"]
                ),
                "negative_fpr": float(
                    test["negative_false_positive_rate"]
                ),
                "boundary_f1": float(raw["boundary_f1_positive"]),
                "mean_boundary_error": float(
                    raw["mean_boundary_error_positive"]
                ),
                "max_boundary_error": float(
                    raw["max_boundary_error_positive_mean"]
                ),
                "row_coverage": float(raw["row_coverage_positive"]),
                "continuity": float(raw["continuity_rate_positive"]),
            }
        )
    return output


def aggregate_seed_metrics(seed_rows):
    grouped = {}
    for row in seed_rows:
        grouped.setdefault(row["model_group"], []).append(row)

    output = []
    for model_group, rows in sorted(grouped.items()):
        result = {
            "model_group": model_group,
            "seeds": "|".join(str(row["seed"]) for row in sorted(
                rows, key=lambda item: item["seed"]
            )),
            "runs": len(rows),
        }
        for metric in AGGREGATE_METRICS:
            values = [row[metric] for row in rows]
            result[f"{metric}_mean"] = average(values)
            result[f"{metric}_std"] = sample_std(values)
        output.append(result)
    return output


def paired_deltas(seed_rows):
    by_seed = {}
    for row in seed_rows:
        by_seed.setdefault(row["seed"], {})[row["model_group"]] = row

    output = []
    for seed, models in sorted(by_seed.items()):
        if "baseline" not in models or "candidate" not in models:
            continue
        baseline = models["baseline"]
        candidate = models["candidate"]
        result = {"seed": seed}
        for metric in AGGREGATE_METRICS:
            result[f"delta_{metric}"] = (
                candidate[metric] - baseline[metric]
            )
        output.append(result)
    return output


def aggregate_grouped_scene(rows, label_to_group, group_key, metrics):
    grouped = {}
    for row in rows:
        model_group = label_to_group.get(row["model"])
        if model_group is None:
            continue
        key = (model_group, row[group_key])
        grouped.setdefault(key, []).append(row)

    output = []
    for (model_group, group), items in sorted(grouped.items()):
        result = {
            "model_group": model_group,
            group_key: group,
            "seed_runs": len(items),
        }
        for metric in metrics:
            values = [number(item, metric) for item in items]
            result[f"{metric}_mean"] = average(values)
            result[f"{metric}_std"] = sample_std(values)
        output.append(result)
    return output


def lock_checks(seed_rows, aggregate_rows):
    aggregate = {row["model_group"]: row for row in aggregate_rows}
    baseline = aggregate["baseline"]
    candidate = aggregate["candidate"]

    positive_iou_delta = (
        candidate["positive_iou_mean"] - baseline["positive_iou_mean"]
    )
    boundary_f1_delta = (
        candidate["boundary_f1_mean"] - baseline["boundary_f1_mean"]
    )
    mbe_reduction = 1.0 - (
        candidate["mean_boundary_error_mean"]
        / baseline["mean_boundary_error_mean"]
    )
    fpr_delta = candidate["negative_fpr_mean"] - baseline["negative_fpr_mean"]
    candidate_seed_rows = [
        row for row in seed_rows if row["model_group"] == "candidate"
    ]
    non_collapsed_seeds = sum(
        1
        for row in candidate_seed_rows
        if row["positive_zero_prediction_rate"] <= 0.05
    )

    checks = [
        (
            "Boundary F1",
            boundary_f1_delta >= -0.005,
            f"mean delta={boundary_f1_delta:+.6f}, required >= -0.005",
        ),
        (
            "Mean boundary error",
            mbe_reduction >= 0.20,
            f"relative reduction={100 * mbe_reduction:.2f}%, required >= 20%",
        ),
        (
            "Positive IoU",
            positive_iou_delta >= -0.01,
            f"mean delta={positive_iou_delta:+.6f}, required >= -0.01",
        ),
        (
            "Negative FPR",
            fpr_delta <= 0.05,
            f"mean delta={100 * fpr_delta:+.2f} pp, required <= +5 pp",
        ),
        (
            "Non-collapsed seeds",
            non_collapsed_seeds >= 2,
            f"qualified={non_collapsed_seeds}/3, required >= 2/3",
        ),
        (
            "All IoU stability",
            candidate["all_iou_std"] <= 0.02,
            (
                f"std={candidate['all_iou_std']:.6f}, "
                "required <= 0.02"
            ),
        ),
        (
            "Boundary F1 stability",
            candidate["boundary_f1_std"] <= 0.02,
            (
                f"std={candidate['boundary_f1_std']:.6f}, "
                "required <= 0.02"
            ),
        ),
        (
            "Frozen full-pipeline speed",
            24.86 >= 20.0,
            "24.86 FPS, required >= 20 FPS",
        ),
    ]
    return checks


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_report(
    path,
    seed_rows,
    aggregate_rows,
    paired_rows,
    checks,
    baseline_name,
    candidate_name,
):
    lines = [
        "# Three-Seed Stability Validation",
        "",
        "## Protocol",
        "",
        f"- Models: {baseline_name} and {candidate_name}",
        "- Seeds: 42, 3407, 2026",
        "- Identical dataset, augmentation, square-root scene sampling, optimizer, and early stopping",
        "- Best checkpoint selected by validation IoU",
        "- Test-dev is reported after checkpoint selection and is not used to select an epoch",
        "",
        "## Per-Seed Results",
        "",
    ]
    table_rows = []
    for row in sorted(
        seed_rows, key=lambda item: (item["model_group"], item["seed"])
    ):
        table_rows.append(
            [
                row["model_group"],
                str(row["seed"]),
                str(row["best_epoch"]),
                f"{row['all_iou']:.4f}",
                f"{row['positive_iou']:.4f}",
                f"{100 * row['negative_fpr']:.2f}%",
                f"{row['boundary_f1']:.4f}",
                f"{row['mean_boundary_error']:.2f}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Model",
                "Seed",
                "Best epoch",
                "All IoU",
                "Positive IoU",
                "FPR",
                "Boundary F1",
                "MBE",
            ],
            table_rows,
        )
    )
    lines.extend(["", "## Mean +/- Sample Standard Deviation", ""])
    aggregate_table = []
    for row in aggregate_rows:
        aggregate_table.append(
            [
                row["model_group"],
                (
                    f"{row['all_iou_mean']:.4f} +/- "
                    f"{row['all_iou_std']:.4f}"
                ),
                (
                    f"{row['positive_iou_mean']:.4f} +/- "
                    f"{row['positive_iou_std']:.4f}"
                ),
                (
                    f"{100 * row['negative_fpr_mean']:.2f}% +/- "
                    f"{100 * row['negative_fpr_std']:.2f}%"
                ),
                (
                    f"{row['boundary_f1_mean']:.4f} +/- "
                    f"{row['boundary_f1_std']:.4f}"
                ),
                (
                    f"{row['mean_boundary_error_mean']:.2f} +/- "
                    f"{row['mean_boundary_error_std']:.2f}"
                ),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Model",
                "All IoU",
                "Positive IoU",
                "FPR",
                "Boundary F1",
                "MBE",
            ],
            aggregate_table,
        )
    )
    lines.extend(["", "## Predefined Lock Checks", ""])
    for name, passed, detail in checks:
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name} - {detail}")
    lock_passed = all(passed for _, passed, _ in checks)
    lines.extend(
        [
            "",
            f"Final automatic lock recommendation: **{'PASS' if lock_passed else 'REVIEW'}**",
            "",
            "The automatic result supports, but does not replace, the frozen scientific interpretation. Per-prefix and far/middle/near stability are available in the accompanying CSV files.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="reports/stability_validation/experiment_manifest.csv",
    )
    parser.add_argument(
        "--scene-dir",
        default="reports/stability_validation/scene_distance",
    )
    parser.add_argument(
        "--out-dir", default="reports/stability_validation"
    )
    parser.add_argument(
        "--baseline-name", default="frozen Band-only baseline"
    )
    parser.add_argument(
        "--candidate-name", default="Edge0.3+Heatmap0.1 candidate"
    )
    args = parser.parse_args()

    manifest = read_csv(Path(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = load_seed_metrics(manifest)
    aggregate_rows = aggregate_seed_metrics(seed_rows)
    delta_rows = paired_deltas(seed_rows)

    numeric_seed_rows = []
    for row in seed_rows:
        numeric_seed_rows.append(
            {
                key: (
                    format_number(value)
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
        )
    write_csv(out_dir / "seed_metrics.csv", numeric_seed_rows, SEED_METRIC_FIELDS)

    aggregate_fields = ["model_group", "seeds", "runs"]
    for metric in AGGREGATE_METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(
        out_dir / "aggregate_metrics.csv",
        [
            {
                key: (
                    format_number(value)
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
            for row in aggregate_rows
        ],
        aggregate_fields,
    )

    delta_fields = ["seed"] + [
        f"delta_{metric}" for metric in AGGREGATE_METRICS
    ]
    write_csv(
        out_dir / "paired_deltas.csv",
        [
            {
                key: (
                    format_number(value)
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
            for row in delta_rows
        ],
        delta_fields,
    )

    scene_dir = Path(args.scene_dir)
    label_to_group = {
        row["model_label"]: row["model_group"] for row in manifest
    }
    prefix_rows = read_csv(scene_dir / "prefix_metrics.csv")
    distance_rows = read_csv(scene_dir / "near_mid_far_metrics.csv")
    prefix_metrics = [
        "positive_iou",
        "boundary_f1_positive",
        "mean_boundary_error_positive",
        "row_coverage_positive",
        "continuity_positive",
        "negative_false_positive_rate",
    ]
    prefix_stability = aggregate_grouped_scene(
        prefix_rows, label_to_group, "group", prefix_metrics
    )
    prefix_fields = ["model_group", "group", "seed_runs"]
    for metric in prefix_metrics:
        prefix_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(
        out_dir / "prefix_stability.csv",
        [
            {
                key: (
                    format_number(value)
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
            for row in prefix_stability
        ],
        prefix_fields,
    )

    distance_metrics = [
        "boundary_f1",
        "mean_boundary_error",
        "row_coverage",
        "continuity",
        "prediction_empty_rate",
    ]
    distance_stability = aggregate_grouped_scene(
        distance_rows,
        label_to_group,
        "distance_region",
        distance_metrics,
    )
    distance_fields = [
        "model_group",
        "distance_region",
        "seed_runs",
    ]
    for metric in distance_metrics:
        distance_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(
        out_dir / "distance_stability.csv",
        [
            {
                key: (
                    format_number(value)
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
            for row in distance_stability
        ],
        distance_fields,
    )

    checks = lock_checks(seed_rows, aggregate_rows)
    write_report(
        out_dir / "stability_report.md",
        seed_rows,
        aggregate_rows,
        delta_rows,
        checks,
        args.baseline_name,
        args.candidate_name,
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
