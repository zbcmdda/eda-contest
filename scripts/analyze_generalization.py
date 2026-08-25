#!/usr/bin/env python3
"""Audit dataset coverage and model errors across architecture-relevant groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--train-rows", type=int, default=900_000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-groups", type=Path, required=True)
    parser.add_argument("--minimum-group-rows", type=int, default=100)
    return parser.parse_args()


def endpoints(series: pd.Series, prefix: str) -> pd.DataFrame:
    parts = series.str.extract(r"^SRB_(\d+)_(\d+)/(.*)$")
    if parts.isna().any().any():
        raise ValueError(f"malformed {prefix} endpoint")
    return pd.DataFrame(
        {
            f"{prefix}_x": parts[0].astype(np.int16),
            f"{prefix}_y": parts[1].astype(np.int16),
            f"{prefix}_port": parts[2].astype("string"),
        },
        index=series.index,
    )


def metric_values(golden: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    positive = golden > 0
    if not positive.any():
        return {"rows": int(len(golden)), "positive_rows": 0}
    relative = np.abs(predicted[positive] - golden[positive]) / golden[positive]
    return {
        "rows": int(len(golden)),
        "positive_rows": int(positive.sum()),
        "official_score": float(np.mean(1.0 - np.tanh(4.0 * relative)) * 100.0),
        "mean_relative_error": float(np.mean(relative)),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "within_5_percent": float(np.mean(relative <= 0.05) * 100.0),
        "within_10_percent": float(np.mean(relative <= 0.10) * 100.0),
        "within_20_percent": float(np.mean(relative <= 0.20) * 100.0),
    }


def architecture_groups(frame: pd.DataFrame, arch: Path) -> pd.DataFrame:
    source = endpoints(frame["From"], "source")
    target = endpoints(frame["To"], "target")
    result = pd.concat([source, target], axis=1)
    dx = target["target_x"].astype(np.int16) - source["source_x"].astype(np.int16)
    dy = target["target_y"].astype(np.int16) - source["source_y"].astype(np.int16)
    result["manhattan"] = dx.abs().astype(np.int16) + dy.abs().astype(np.int16)
    result["distance_bin"] = pd.cut(
        result["manhattan"],
        bins=[-1, 16, 32, 64, 128, 256, 384, 512, np.inf],
        labels=["0-16", "17-32", "33-64", "65-128", "129-256", "257-384", "385-512", "513+"],
    ).astype("string")
    result["source_kind"] = result["source_port"].str[0]
    result["target_kind"] = result["target_port"].str[0]
    result["port_kind_pair"] = result["source_kind"] + "->" + result["target_kind"]
    result["source_region"] = (
        (result["source_x"] // 20).astype("string") + ":" + (result["source_y"] // 50).astype("string")
    )
    result["target_region"] = (
        (result["target_x"] // 20).astype("string") + ":" + (result["target_y"] // 50).astype("string")
    )

    with (arch / "SRB_Gap.json").open(encoding="utf-8") as stream:
        gaps = json.load(stream)["Gap"]
    line_delay = np.zeros(len(frame), dtype=np.int32)
    for line in gaps["Line"]:
        if line["direction"] == "vertical":
            crossed = (
                np.minimum(result["source_x"], result["target_x"]) <= line["site"]
            ) & (line["site"] < np.maximum(result["source_x"], result["target_x"]))
        else:
            crossed = (
                np.minimum(result["source_y"], result["target_y"]) <= line["site"]
            ) & (line["site"] < np.maximum(result["source_y"], result["target_y"]))
        line_delay[crossed.to_numpy()] += int(line["delay"])
    result["line_delay_bin"] = pd.cut(
        line_delay,
        bins=[-1, 0, 25, 75, 150, np.inf],
        labels=["0", "1-25", "26-75", "76-150", "151+"],
    ).astype("string")

    block_relation = np.full(len(frame), "none", dtype=object)
    for block in gaps["Block"]:
        same_horizontal_band = (
            result["source_y"].between(block["lower"], block["upper"])
            & result["target_y"].between(block["lower"], block["upper"])
        )
        crosses_horizontal = (
            ((result["source_x"] < block["left"]) & (result["target_x"] > block["right"]))
            | ((result["source_x"] > block["right"]) & (result["target_x"] < block["left"]))
        )
        same_vertical_band = (
            result["source_x"].between(block["left"], block["right"])
            & result["target_x"].between(block["left"], block["right"])
        )
        crosses_vertical = (
            ((result["source_y"] < block["lower"]) & (result["target_y"] > block["upper"]))
            | ((result["source_y"] > block["upper"]) & (result["target_y"] < block["lower"]))
        )
        selected = (same_horizontal_band & crosses_horizontal) | (same_vertical_band & crosses_vertical)
        block_relation[selected.to_numpy()] = f"block_{block['id']}"
    result["block_relation"] = pd.Series(block_relation, dtype="string")
    return result


def coverage(frame: pd.DataFrame, groups: pd.DataFrame) -> dict[str, int]:
    pairs = frame["From"] + "\0" + frame["To"]
    port_pairs = groups["source_port"] + "\0" + groups["target_port"]
    return {
        "rows": int(len(frame)),
        "unique_pairs": int(pairs.nunique()),
        "unique_sources": int(frame["From"].nunique()),
        "unique_targets": int(frame["To"].nunique()),
        "unique_source_ports": int(groups["source_port"].nunique()),
        "unique_target_ports": int(groups["target_port"].nunique()),
        "unique_port_pairs": int(port_pairs.nunique()),
        "source_x_values": int(groups["source_x"].nunique()),
        "source_y_values": int(groups["source_y"].nunique()),
        "target_x_values": int(groups["target_x"].nunique()),
        "target_y_values": int(groups["target_y"].nunique()),
    }


def main() -> None:
    args = parse_args()
    answers = pd.read_csv(args.answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    predictions = pd.read_csv(
        args.predictions, dtype={"From": "string", "To": "string", "delay": np.int32}
    )
    if len(answers) != len(predictions):
        raise ValueError("answer and prediction row counts differ")
    if not answers[["From", "To"]].equals(predictions[["From", "To"]]):
        raise ValueError("answer and prediction endpoints differ")
    if args.train_rows <= 0 or args.train_rows >= len(answers):
        raise ValueError("train row count must split the dataset")

    groups = architecture_groups(answers, args.arch)
    golden = answers["delay"].to_numpy(dtype=np.float64)
    predicted = predictions["delay"].to_numpy(dtype=np.float64)
    split_masks = {
        "train": np.arange(len(answers)) < args.train_rows,
        "validation": np.arange(len(answers)) >= args.train_rows,
    }

    report: dict[str, object] = {"splits": {}, "coverage": {}}
    for name, mask in split_masks.items():
        report["splits"][name] = metric_values(golden[mask], predicted[mask])
        report["coverage"][name] = coverage(answers.loc[mask], groups.loc[mask])
    train_pairs = set(zip(answers.loc[split_masks["train"], "From"], answers.loc[split_masks["train"], "To"]))
    valid_pairs = set(
        zip(answers.loc[split_masks["validation"], "From"], answers.loc[split_masks["validation"], "To"])
    )
    report["exact_pair_overlap"] = len(train_pairs & valid_pairs)

    validation = split_masks["validation"]
    rows: list[dict[str, object]] = []
    group_columns = [
        "distance_bin",
        "source_kind",
        "target_kind",
        "port_kind_pair",
        "source_region",
        "target_region",
        "line_delay_bin",
        "block_relation",
        "source_port",
        "target_port",
    ]
    for column in group_columns:
        values = groups.loc[validation, column]
        for value, index in values.groupby(values, observed=True).groups.items():
            positions = np.asarray(index, dtype=np.int64)
            if len(positions) < args.minimum_group_rows:
                continue
            metrics = metric_values(golden[positions], predicted[positions])
            rows.append({"group_type": column, "group": str(value), **metrics})
    grouped = pd.DataFrame(rows).sort_values(["group_type", "official_score", "group"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_groups.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    grouped.to_csv(args.output_groups, index=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {len(grouped)} group rows to {args.output_groups}")


if __name__ == "__main__":
    main()
