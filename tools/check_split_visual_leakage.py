import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path):
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def sample_id(row):
    return row.get("sample_id") or Path(row["image"]).stem


def main():
    parser = argparse.ArgumentParser(
        description="Screen train/val/test for exact and near-duplicate images."
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/emergency_outer_boundary_grouped_fixed12_v2_clean601",
    )
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--out-dir", default="reports/split_visual_leakage")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    records = {}
    for split in ("train", "val", "test"):
        rows = read_rows(root / "splits" / f"{split}.csv")
        split_records = []
        for row in rows:
            path = root / row["image"]
            if not path.is_file():
                raise FileNotFoundError(path)
            split_records.append(
                {
                    "split": split,
                    "sample_id": sample_id(row),
                    "image": row["image"],
                    "sha256": file_sha256(path),
                    "dhash": dhash64(path),
                }
            )
        records[split] = split_records

    candidates = []
    pair_summaries = []
    for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
        left_rows = records[left_name]
        right_rows = records[right_name]
        exact = 0
        near = 0
        minimum = 64
        for left in left_rows:
            for right in right_rows:
                distance = (left["dhash"] ^ right["dhash"]).bit_count()
                minimum = min(minimum, distance)
                is_exact = left["sha256"] == right["sha256"]
                if is_exact:
                    exact += 1
                if distance <= args.threshold:
                    near += 1
                    candidates.append(
                        {
                            "left_split": left_name,
                            "right_split": right_name,
                            "left_sample_id": left["sample_id"],
                            "right_sample_id": right["sample_id"],
                            "left_image": left["image"],
                            "right_image": right["image"],
                            "exact_sha256": int(is_exact),
                            "dhash_hamming": distance,
                        }
                    )
        pair_summaries.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "left_images": len(left_rows),
                "right_images": len(right_rows),
                "exact_pairs": exact,
                "near_candidate_pairs": near,
                "minimum_dhash_hamming": minimum,
            }
        )

    candidates.sort(
        key=lambda row: (
            row["dhash_hamming"],
            row["left_split"],
            row["left_sample_id"],
        )
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidate_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "left_split",
            "right_split",
            "left_sample_id",
            "right_sample_id",
            "left_image",
            "right_image",
            "exact_sha256",
            "dhash_hamming",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    with (out_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_summaries[0]))
        writer.writeheader()
        writer.writerows(pair_summaries)

    lines = [
        "# Split Visual Leakage Screening",
        "",
        f"- Dataset: `{root}`",
        f"- Near-duplicate candidate threshold: dHash Hamming <= {args.threshold}",
        "- SHA-256 equality proves exact file duplication.",
        "- dHash is only a screening heuristic; candidates require manual review.",
        "",
        "| Split pair | Images | Exact pairs | Near candidates | Minimum dHash distance |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pair_summaries:
        lines.append(
            f"| {row['left_split']}/{row['right_split']} | "
            f"{row['left_images']} x {row['right_images']} | "
            f"{row['exact_pairs']} | {row['near_candidate_pairs']} | "
            f"{row['minimum_dhash_hamming']} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
