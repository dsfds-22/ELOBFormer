import argparse
import csv
import json
import math
import random
from pathlib import Path


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_mcnemar_p(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def cluster_bootstrap(pairs, iterations, seed):
    by_cluster = {}
    for pair in pairs:
        by_cluster.setdefault(pair["scene_group"], []).append(pair)
    clusters = sorted(by_cluster)
    rng = random.Random(seed)
    cmbe_differences = []
    bf1_differences = []
    for _ in range(iterations):
        sampled = []
        for _ in clusters:
            sampled.extend(by_cluster[rng.choice(clusters)])
        cmbe_differences.append(
            mean(item["right_cmbe"] - item["left_cmbe"] for item in sampled)
        )
        bf1_differences.append(
            mean(item["right_bf1"] - item["left_bf1"] for item in sampled)
        )
    return {
        "clusters": len(clusters),
        "iterations": iterations,
        "seed": seed,
        "cmbe_difference_ci95": [
            percentile(cmbe_differences, 0.025),
            percentile(cmbe_differences, 0.975),
        ],
        "bf1_difference_ci95": [
            percentile(bf1_differences, 0.025),
            percentile(bf1_differences, 0.975),
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Paired raw-mask boundary evaluation with cluster bootstrap."
    )
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--right-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_rows(args.metrics_csv)
    left = {
        row["sample_id"]: row
        for row in rows
        if row["model"] == args.left_model
    }
    right = {
        row["sample_id"]: row
        for row in rows
        if row["model"] == args.right_model
    }
    sample_ids = sorted(set(left) & set(right))
    positive_ids = [
        sample_id
        for sample_id in sample_ids
        if int(left[sample_id]["has_boundary"]) == 1
        and int(right[sample_id]["has_boundary"]) == 1
    ]
    negative_ids = [
        sample_id
        for sample_id in sample_ids
        if int(left[sample_id]["has_boundary"]) == 0
        and int(right[sample_id]["has_boundary"]) == 0
    ]

    both_nonempty = []
    left_empty_right_nonempty = []
    left_nonempty_right_empty = []
    both_empty = []
    pair_rows = []
    for sample_id in positive_ids:
        left_row = left[sample_id]
        right_row = right[sample_id]
        left_empty = int(left_row["prediction_empty"]) == 1
        right_empty = int(right_row["prediction_empty"]) == 1
        if left_empty and right_empty:
            category = "both_empty"
            both_empty.append(sample_id)
        elif left_empty:
            category = "left_empty_right_nonempty"
            left_empty_right_nonempty.append(sample_id)
        elif right_empty:
            category = "left_nonempty_right_empty"
            left_nonempty_right_empty.append(sample_id)
        else:
            category = "both_nonempty"
            both_nonempty.append(
                {
                    "sample_id": sample_id,
                    "scene_group": left_row["scene_group"],
                    "left_cmbe": float(left_row["conditional_mean_boundary_error"]),
                    "right_cmbe": float(right_row["conditional_mean_boundary_error"]),
                    "left_bf1": float(left_row["boundary_f1"]),
                    "right_bf1": float(right_row["boundary_f1"]),
                }
            )
        pair_rows.append(
            {
                "sample_id": sample_id,
                "prefix": left_row["prefix"],
                "scene_group": left_row["scene_group"],
                "category": category,
                "left_prediction_empty": int(left_empty),
                "right_prediction_empty": int(right_empty),
                "left_bf1": left_row["boundary_f1"],
                "right_bf1": right_row["boundary_f1"],
                "left_cmbe": left_row["conditional_mean_boundary_error"],
                "right_cmbe": right_row["conditional_mean_boundary_error"],
            }
        )

    negative_left_fp_right_tn = 0
    negative_left_tn_right_fp = 0
    for sample_id in negative_ids:
        left_fp = int(left[sample_id]["prediction_empty"]) == 0
        right_fp = int(right[sample_id]["prediction_empty"]) == 0
        negative_left_fp_right_tn += int(left_fp and not right_fp)
        negative_left_tn_right_fp += int(not left_fp and right_fp)

    recovered_right_rows = [right[sample_id] for sample_id in left_empty_right_nonempty]
    bootstrap = cluster_bootstrap(
        both_nonempty, args.bootstrap_iterations, args.seed
    )
    summary = {
        "protocol": {
            "metrics_csv": str(Path(args.metrics_csv)),
            "left_model": args.left_model,
            "right_model": args.right_model,
            "comparison": "right_minus_left",
        },
        "positive_pairs": len(positive_ids),
        "negative_pairs": len(negative_ids),
        "both_nonempty": len(both_nonempty),
        "left_empty_right_nonempty": len(left_empty_right_nonempty),
        "left_nonempty_right_empty": len(left_nonempty_right_empty),
        "both_empty": len(both_empty),
        "positive_empty_mcnemar_p": exact_mcnemar_p(
            len(left_empty_right_nonempty), len(left_nonempty_right_empty)
        ),
        "common_nonempty": {
            "left_cmbe": mean(item["left_cmbe"] for item in both_nonempty),
            "right_cmbe": mean(item["right_cmbe"] for item in both_nonempty),
            "cmbe_difference_right_minus_left": mean(
                item["right_cmbe"] - item["left_cmbe"]
                for item in both_nonempty
            ),
            "left_bf1": mean(item["left_bf1"] for item in both_nonempty),
            "right_bf1": mean(item["right_bf1"] for item in both_nonempty),
            "bf1_difference_right_minus_left": mean(
                item["right_bf1"] - item["left_bf1"]
                for item in both_nonempty
            ),
        },
        "right_recovered_samples": {
            "count": len(recovered_right_rows),
            "sample_ids": left_empty_right_nonempty,
            "mean_cmbe": mean(
                float(row["conditional_mean_boundary_error"])
                for row in recovered_right_rows
            ),
            "mean_bf1": mean(
                float(row["boundary_f1"]) for row in recovered_right_rows
            ),
        },
        "right_new_misses": left_nonempty_right_empty,
        "negative_discordance": {
            "left_fp_right_tn": negative_left_fp_right_tn,
            "left_tn_right_fp": negative_left_tn_right_fp,
            "mcnemar_p": exact_mcnemar_p(
                negative_left_fp_right_tn, negative_left_tn_right_fp
            ),
        },
        "cluster_bootstrap": bootstrap,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "paired_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    common = summary["common_nonempty"]
    cmbe_ci = bootstrap["cmbe_difference_ci95"]
    bf1_ci = bootstrap["bf1_difference_ci95"]
    report = [
        "# Paired Boundary Evaluation",
        "",
        "## Protocol",
        "",
        f"- Left: `{args.left_model}`",
        f"- Right: `{args.right_model}`",
        "- All differences are Right - Left.",
        "- CMBE/BF1 pairing uses only positive samples where both models predict non-empty masks.",
        "- Confidence intervals use scene-group cluster bootstrap.",
        "",
        "## Positive Detection Pairing",
        "",
        f"- Positive pairs: {len(positive_ids)}",
        f"- Both non-empty: {len(both_nonempty)}",
        f"- Left empty, right non-empty: {len(left_empty_right_nonempty)}",
        f"- Left non-empty, right empty: {len(left_nonempty_right_empty)}",
        f"- Both empty: {len(both_empty)}",
        f"- Exact two-sided McNemar p: {summary['positive_empty_mcnemar_p']:.6f}",
        "",
        "## Common Non-empty Geometry",
        "",
        f"- Left CMBE: {common['left_cmbe']:.6f} px",
        f"- Right CMBE: {common['right_cmbe']:.6f} px",
        f"- CMBE difference: {common['cmbe_difference_right_minus_left']:.6f} px",
        f"- Cluster-bootstrap 95% CI: [{cmbe_ci[0]:.6f}, {cmbe_ci[1]:.6f}] px",
        f"- Left BF1: {common['left_bf1']:.6f}",
        f"- Right BF1: {common['right_bf1']:.6f}",
        f"- BF1 difference: {common['bf1_difference_right_minus_left']:.6f}",
        f"- Cluster-bootstrap 95% CI: [{bf1_ci[0]:.6f}, {bf1_ci[1]:.6f}]",
        "",
        "## Recovered And New-miss Samples",
        "",
        f"- Right recovered: {len(left_empty_right_nonempty)}",
        f"- Recovered mean CMBE: {summary['right_recovered_samples']['mean_cmbe']:.6f} px",
        f"- Recovered mean BF1: {summary['right_recovered_samples']['mean_bf1']:.6f}",
        f"- Right new misses: {len(left_nonempty_right_empty)}",
        "",
        "## Negative Pairing",
        "",
        f"- Left false positive, right true negative: {negative_left_fp_right_tn}",
        f"- Left true negative, right false positive: {negative_left_tn_right_fp}",
        f"- Exact two-sided McNemar p: {summary['negative_discordance']['mcnemar_p']:.6f}",
    ]
    (out_dir / "paired_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
