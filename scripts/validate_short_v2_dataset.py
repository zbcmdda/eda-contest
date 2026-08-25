#!/usr/bin/env python3
"""Check short-v2 candidate invariants without computing any model metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--existing-candidates", type=Path, action="append", default=[])
    parser.add_argument("--public-labeled", type=Path)
    parser.add_argument("--public-expected", type=Path)
    parser.add_argument("--label-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_endpoint(series: pd.Series) -> pd.DataFrame:
    result = series.str.extract(r"^SRB_(\d+)_(\d+)/(.*)$")
    if result.isna().any().any():
        raise ValueError("malformed endpoint")
    return result.rename(columns={0: "x", 1: "y", 2: "port"}).astype({"x": np.int16, "y": np.int16})


def pair_keys(frame: pd.DataFrame) -> pd.Series:
    return frame["From"].astype("string") + "\0" + frame["To"].astype("string")


def candidate_files(args: argparse.Namespace) -> list[Path]:
    discovered = sorted(Path("refine-logs").glob("*candidates*.csv"))
    return sorted({path.resolve() for path in [*discovered, *args.existing_candidates]
                   if path.exists() and not path.name.endswith(".metadata.csv")})


def main() -> None:
    args = parse_args()
    frames = []
    for split in ("dev", "sealed_iid", "sealed_ood"):
        metadata_path = args.dataset_dir / f"{split}.metadata.csv"
        endpoint_path = args.dataset_dir / f"{split}.csv"
        metadata = pd.read_csv(metadata_path, dtype="string")
        endpoints = pd.read_csv(endpoint_path, dtype="string")
        if not metadata[["From", "To"]].equals(endpoints[["From", "To"]]):
            raise ValueError(f"{split} endpoint file does not match metadata")
        if set(metadata["split"]) != {split}:
            raise ValueError(f"{split} metadata has incorrect split column")
        frames.append(metadata)
    frame = pd.concat(frames, ignore_index=True)
    key = pair_keys(frame)
    if key.duplicated().any():
        raise ValueError("duplicate endpoint pair inside short-v2 dataset")

    source, target = parse_endpoint(frame["From"]), parse_endpoint(frame["To"])
    dx = target["x"].astype(np.int16) - source["x"].astype(np.int16)
    dy = target["y"].astype(np.int16) - source["y"].astype(np.int16)
    distance = dx.abs() + dy.abs()
    strata = pd.cut(distance, bins=[-1, 0, 4, 8, 16, 24, 32],
                    labels=["0", "1-4", "5-8", "9-16", "17-24", "25-32"]).astype("string")
    if not strata.equals(frame["stratum"]):
        raise ValueError("metadata stratum does not equal endpoint Manhattan distance")
    if not (dx.astype("string") == frame["dx"]).all() or not (dy.astype("string") == frame["dy"]).all():
        raise ValueError("metadata dx/dy does not equal endpoints")
    expected_groups = [json.dumps([sp, tp, int(x), int(y)], separators=(",", ":"))
                       for sp, tp, x, y in zip(source["port"], target["port"], dx, dy)]
    if expected_groups != frame["group_key"].tolist():
        raise ValueError("group_key does not equal (source_port,target_port,dx,dy)")
    group_splits = frame.groupby("group_key", observed=True)["split"].nunique()
    if (group_splits > 1).any():
        raise ValueError("a group_key leaks across split boundaries")
    if not (frame.loc[frame["split"] == "sealed_ood", "ood_stress"].astype(int) == 1).all():
        raise ValueError("sealed_ood contains a non-stress endpoint")

    excluded = set(pair_keys(pd.read_csv(args.answers, usecols=["From", "To"], dtype="string")))
    for path in candidate_files(args):
        existing = pd.read_csv(path, usecols=["From", "To"], dtype="string")
        excluded.update(pair_keys(existing))
    overlap = int(key.isin(excluded).sum())
    if overlap:
        raise ValueError(f"short-v2 overlaps {overlap} original/candidate endpoint pairs")

    oracle = None
    if args.public_labeled or args.public_expected:
        if not args.public_labeled or not args.public_expected:
            raise ValueError("public label comparison requires both --public-labeled and --public-expected")
        actual = pd.read_csv(args.public_labeled, dtype={"From": "string", "To": "string", "delay": np.int64})
        expected = pd.read_csv(args.public_expected, dtype={"From": "string", "To": "string", "delay": np.int64})
        if not actual[["From", "To"]].equals(expected[["From", "To"]]):
            raise ValueError("public exact endpoints do not match expected endpoints")
        mismatch = int((actual["delay"] != expected["delay"]).sum())
        if mismatch:
            raise ValueError(f"public exact oracle mismatches: {mismatch}")
        by_stratum = {}
        for stratum, positions in expected.groupby("stratum", observed=True).groups.items():
            position = np.asarray(list(positions), dtype=np.int64)
            by_stratum[str(stratum)] = {
                "rows": int(len(position)),
                "mismatches": int((actual["delay"].iloc[position] != expected["delay"].iloc[position]).sum()),
            }
        oracle = {"rows": int(len(actual)), "mismatches": mismatch, "by_stratum": by_stratum}

    label_reports = []
    for path in args.label_report:
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report.get("unreachable_rows", -1)) != 0:
            raise ValueError(f"nonzero unreachable rows in {path}")
        label_reports.append({"path": str(path), "input_rows": report.get("input_rows"),
                              "unreachable_rows": report.get("unreachable_rows"),
                              "output_sha256": report.get("output_sha256")})
    report = {
        "rows": int(len(frame)), "unique_pairs": int(key.nunique()), "overlap_pairs": overlap,
        "split_rows": frame["split"].value_counts().sort_index().to_dict(),
        "stratum_rows": frame.groupby(["split", "stratum"]).size().unstack(fill_value=0).to_dict(),
        "unique_group_keys": int(frame["group_key"].nunique()), "group_split_leaks": 0,
        "public_exact": oracle, "label_reports": label_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
