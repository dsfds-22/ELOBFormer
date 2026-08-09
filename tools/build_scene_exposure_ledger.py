from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601"
SPLIT_DIR = DATASET_ROOT / "splits"
ERROR_ANALYSIS_CSV = (
    ROOT / "reports/unified_mask_evaluation_v2_clean601/per_image_metrics.csv"
)
GATE_DESIGN_CSV = ROOT / "reports/v4_3_band_only_deploy_v2_clean601/image_summary.csv"
OUTPUT_DIR = ROOT / "reports/scene_exposure_audit_v2_clean601"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def effective_scene_group(row: dict[str, str]) -> str:
    return row.get("reviewed_segment_id", "").strip() or row["scene_segment"].strip()


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    split_rows: list[dict[str, str]] = []
    sample_to_group: dict[str, str] = {}
    for split_name, filename in (
        ("train", "train.csv"),
        ("validation", "val.csv"),
        ("test_dev", "test_dev_clean601.csv"),
    ):
        for row in read_csv(SPLIT_DIR / filename):
            row = dict(row)
            row["_split_name"] = split_name
            row["_scene_group"] = effective_scene_group(row)
            split_rows.append(row)
            sample_to_group[row["sample_id"]] = row["_scene_group"]

    # The unified clean601 per-image table contains several model rows for each
    # Test-dev image. Presence in this table means the scene was exposed during
    # quantitative error analysis, irrespective of whether every frame was
    # manually inspected.
    error_analysis_samples = {
        row["sample_id"] for row in read_csv(ERROR_ANALYSIS_CSV)
    }

    # Test-dev was iteratively used during V2/V3/V4.3 development. We therefore
    # conservatively classify every scene represented in the final full-set gate
    # run as exposed to gate design/tuning, not merely final evaluation.
    gate_design_samples = {row["sample_id"] for row in read_csv(GATE_DESIGN_CSV)}

    unknown_error_samples = sorted(error_analysis_samples - sample_to_group.keys())
    unknown_gate_samples = sorted(gate_design_samples - sample_to_group.keys())
    if unknown_error_samples or unknown_gate_samples:
        raise RuntimeError(
            "Exposure evidence contains samples absent from the frozen split: "
            f"error={unknown_error_samples[:5]}, gate={unknown_gate_samples[:5]}"
        )

    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in split_rows:
        group_rows[row["_scene_group"]].append(row)

    ledger: list[dict[str, object]] = []
    for group_name in sorted(group_rows):
        rows = group_rows[group_name]
        split_names = {row["_split_name"] for row in rows}
        if len(split_names) != 1:
            raise RuntimeError(f"Scene group crosses splits: {group_name} -> {split_names}")

        sample_ids = {row["sample_id"] for row in rows}
        prefixes = sorted({row["prefix"] for row in rows})
        capture_sessions = sorted({row["capture_session"] for row in rows})
        positive = sum(int(row["has_boundary"]) for row in rows)
        negative = len(rows) - positive
        in_train = "train" in split_names
        in_validation = "validation" in split_names
        in_test_dev = "test_dev" in split_names
        in_error_analysis = bool(sample_ids & error_analysis_samples)
        in_gate_design = bool(sample_ids & gate_design_samples)
        completely_unexposed = not any(
            (in_train, in_validation, in_test_dev, in_error_analysis, in_gate_design)
        )

        ledger.append(
            {
                "scene_group": group_name,
                "group_type": "reviewed_segment" if group_name[:3].isdigit() else "independent_session",
                "prefix": "|".join(prefixes),
                "capture_session": "|".join(capture_sessions),
                "image_count": len(rows),
                "positive_count": positive,
                "negative_count": negative,
                "participated_training": yes_no(in_train),
                "participated_validation": yes_no(in_validation),
                "participated_test_dev": yes_no(in_test_dev),
                "participated_error_analysis": yes_no(in_error_analysis),
                "participated_gate_design": yes_no(in_gate_design),
                "completely_unexposed": yes_no(completely_unexposed),
                "error_analysis_image_count": len(sample_ids & error_analysis_samples),
                "gate_design_image_count": len(sample_ids & gate_design_samples),
            }
        )

    if len(ledger) != 87:
        raise RuntimeError(f"Expected 87 scene groups, found {len(ledger)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = OUTPUT_DIR / "scene_exposure_ledger.csv"
    with ledger_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)

    totals = {
        "scene_groups": len(ledger),
        "images": sum(int(row["image_count"]) for row in ledger),
        "positive_images": sum(int(row["positive_count"]) for row in ledger),
        "negative_images": sum(int(row["negative_count"]) for row in ledger),
        "training_scene_groups": sum(row["participated_training"] == "yes" for row in ledger),
        "validation_scene_groups": sum(
            row["participated_validation"] == "yes" for row in ledger
        ),
        "test_dev_scene_groups": sum(
            row["participated_test_dev"] == "yes" for row in ledger
        ),
        "error_analysis_scene_groups": sum(
            row["participated_error_analysis"] == "yes" for row in ledger
        ),
        "gate_design_scene_groups": sum(
            row["participated_gate_design"] == "yes" for row in ledger
        ),
        "completely_unexposed_scene_groups": sum(
            row["completely_unexposed"] == "yes" for row in ledger
        ),
    }
    (OUTPUT_DIR / "scene_exposure_summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# 场景组暴露审计：fixed12_v2_clean601",
        "",
        "## 审计口径",
        "",
        "- 场景键：021-026 使用人工复核后的 `reviewed_segment_id`，其余使用 `scene_segment`。",
        "- 错误分析暴露：该场景至少一张图进入 clean601 统一逐图评价。",
        "- 门控器设计暴露：该场景至少一张图进入 V4.3 全量门控运行。由于 Test-dev 曾用于 V2/V3/V4.3 的迭代分析，保守地将其视为设计/调参暴露，而不仅是最终评价。",
        "- 只有训练、验证、Test-dev、错误分析和门控器设计五项均为否，才算完全未暴露。",
        "",
        "## 汇总",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 场景组 | {totals['scene_groups']} |",
        f"| 图像 | {totals['images']} |",
        f"| 正样本图像 | {totals['positive_images']} |",
        f"| 负样本图像 | {totals['negative_images']} |",
        f"| 参与训练的场景组 | {totals['training_scene_groups']} |",
        f"| 参与验证的场景组 | {totals['validation_scene_groups']} |",
        f"| 参与 Test-dev 的场景组 | {totals['test_dev_scene_groups']} |",
        f"| 参与错误分析的场景组 | {totals['error_analysis_scene_groups']} |",
        f"| 参与门控器设计的场景组 | {totals['gate_design_scene_groups']} |",
        f"| 完全未暴露场景组 | **{totals['completely_unexposed_scene_groups']}** |",
        "",
        "## 全部场景组明细",
        "",
        "| 场景组 | 前缀 | 图像 | 正样本 | 负样本 | 训练 | 验证 | Test-dev | 错误分析 | 门控器设计 | 完全未暴露 |",
        "|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in ledger:
        md_lines.append(
            "| {scene_group} | {prefix} | {image_count} | {positive_count} | "
            "{negative_count} | {train} | {val} | {test} | {error} | {gate} | {unexposed} |".format(
                **row,
                train="是" if row["participated_training"] == "yes" else "否",
                val="是" if row["participated_validation"] == "yes" else "否",
                test="是" if row["participated_test_dev"] == "yes" else "否",
                error="是" if row["participated_error_analysis"] == "yes" else "否",
                gate="是" if row["participated_gate_design"] == "yes" else "否",
                unexposed="是" if row["completely_unexposed"] == "yes" else "否",
            )
        )

    md_lines.extend(
        [
            "",
            "## 决策含义",
            "",
            "现有场景中不存在完全未暴露场景。72个训练场景组和5个验证场景组已直接参与模型开发；10个Test-dev场景组也全部用于重复错误分析和V4.3门控器开发。因此，现有87组不能再构成严格未触碰的Final-Test。",
            "",
            "若要建立严格Final-Test，需要新增独立采集场景组。若无法补采，较稳妥的替代方案是重新进行场景级划分、从头训练并采用分组交叉验证，但这不等价于完全未触碰的Final-Test。",
            "",
            "## 证据来源",
            "",
            f"- Frozen splits: `{SPLIT_DIR.relative_to(ROOT).as_posix()}`",
            f"- Error-analysis table: `{ERROR_ANALYSIS_CSV.relative_to(ROOT).as_posix()}`",
            f"- Gate-design table: `{GATE_DESIGN_CSV.relative_to(ROOT).as_posix()}`",
        ]
    )
    (OUTPUT_DIR / "scene_exposure_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"Saved: {ledger_path}")


if __name__ == "__main__":
    main()
