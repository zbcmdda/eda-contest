#!/usr/bin/env python3
"""Qualify the locked compact 0--128 residual candidate on development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from short_features import NAMES, TWO_HOP_NAMES
from train_model import ArchitectureFeatures, official_metrics


SEGMENTS = (("0-16", 0, 16), ("17-32", 17, 32), ("33-64", 33, 64), ("65-128", 65, 128))
SEEDS = (20260815, 20260816, 20260817, 20260824, 20260825)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def distances(frame: pd.DataFrame) -> np.ndarray:
    source = frame["From"].str.extract(r"^SRB_(\d+)_(\d+)/").astype(np.int16)
    target = frame["To"].str.extract(r"^SRB_(\d+)_(\d+)/").astype(np.int16)
    return ((source[0] - target[0]).abs() + (source[1] - target[1]).abs()).to_numpy(np.int16)


def row_score(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full(len(truth), 100.0)
    selected = truth > 0
    relative = np.abs(prediction[selected] - truth[selected]) / truth[selected]
    result[selected] = (1.0 - np.tanh(4.0 * relative)) * 100.0
    return result


def corrected(baseline: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = np.rint(np.maximum(0.0, (baseline + 1.0) * np.exp(1.5 * residual) - 1.0))
    result[baseline == 0] = 0
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        args.answers,
        nrows=900_000,
        dtype={"From": "string", "To": "string", "delay": np.int32},
    )
    raw["row_id"] = np.arange(len(raw), dtype=np.int32)
    raw["manhattan"] = distances(raw)
    short = raw.loc[raw["manhattan"] <= 128].reset_index(drop=True)
    train = short["row_id"].to_numpy() < 800_000
    development = ~train
    base_features = ArchitectureFeatures(args.arch).build(short[["From", "To", "delay"]])
    local = pd.read_csv(
        args.feature_dir / "short_0_128.features.csv", usecols=NAMES, dtype=np.int32
    )
    features = pd.concat([base_features.reset_index(drop=True), local], axis=1)
    categorical = [
        column for column in base_features
        if isinstance(base_features[column].dtype, pd.CategoricalDtype)
    ]
    baseline = pd.read_csv(
        args.base_dir / "short_0_128.baseline.csv", usecols=["delay"]
    )["delay"].to_numpy(dtype=np.float64)
    truth = short["delay"].to_numpy(dtype=np.float64)
    distance = short["manhattan"].to_numpy(dtype=np.int16)
    row_id = short["row_id"].to_numpy(dtype=np.int32)
    label = np.log((truth[train] + 1.0) / (baseline[train] + 1.0))
    common = {
        "objective": "regression_l1",
        "metric": "l1",
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": args.threads,
        "force_col_wise": True,
    }
    runs = []
    for seed in SEEDS:
        medium_columns = list(base_features.columns) + TWO_HOP_NAMES
        medium = lgb.train(
            {
                **common, "seed": seed, "num_leaves": 1023, "max_depth": 14,
                "learning_rate": 0.2, "min_data_in_leaf": 100,
            },
            lgb.Dataset(
                features.loc[train, medium_columns], label=label,
                categorical_feature=categorical, free_raw_data=False,
            ),
            num_boost_round=16,
        )
        ultra_train = train & (distance <= 32)
        ultra = lgb.train(
            {
                **common, "seed": seed, "num_leaves": 511, "max_depth": 12,
                "learning_rate": 0.08, "min_data_in_leaf": 20,
            },
            lgb.Dataset(
                features.loc[ultra_train], label=label[distance[train] <= 32],
                categorical_feature=categorical, free_raw_data=False,
            ),
            num_boost_round=64,
        )
        medium.save_model(str(args.output_dir / f"medium_seed{seed}.txt"))
        ultra.save_model(str(args.output_dir / f"ultra_seed{seed}.txt"))

        dev_features = features.loc[development]
        dev_truth = truth[development]
        dev_baseline = baseline[development]
        dev_distance = distance[development]
        dev_rows = row_id[development]
        prediction = dev_baseline.copy()
        ultra_rows = dev_distance <= 32
        medium_rows = ~ultra_rows
        prediction[ultra_rows] = corrected(
            dev_baseline[ultra_rows], ultra.predict(dev_features.loc[ultra_rows])
        )
        prediction[medium_rows] = corrected(
            dev_baseline[medium_rows],
            medium.predict(dev_features.loc[medium_rows, medium_columns]),
        )
        candidate_rows = row_score(dev_truth, prediction)
        baseline_rows = row_score(dev_truth, dev_baseline)
        report = {
            "seed": seed,
            "short_score": official_metrics(dev_truth, prediction)["official_score"],
            "short_delta": official_metrics(dev_truth, prediction)["official_score"]
            - official_metrics(dev_truth, dev_baseline)["official_score"],
            "full_development_delta": float(np.sum(candidate_rows - baseline_rows) / 100_000.0),
            "segments": {},
            "contiguous_blocks": {},
        }
        for name, low, high in SEGMENTS:
            selected = (dev_distance >= low) & (dev_distance <= high)
            report["segments"][name] = (
                official_metrics(dev_truth[selected], prediction[selected])["official_score"]
                - official_metrics(dev_truth[selected], dev_baseline[selected])["official_score"]
            )
        for low in (800_000, 825_000, 850_000, 875_000):
            selected = (dev_rows >= low) & (dev_rows < low + 25_000)
            report["contiguous_blocks"][f"{low}-{low + 24_999}"] = float(
                np.sum(candidate_rows[selected] - baseline_rows[selected]) / 25_000.0
            )
        runs.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    summary = {
        "protocol": {
            "training": "public rows 0-799999",
            "development": "public rows 800000-899999",
            "frozen_rows_read": False,
            "sealed_labels_read": False,
            "gate": "0-32 uses depth-3/64xdepth-12; 33-128 uses depth-2/16xdepth-14",
            "residual_multiplier": 1.5,
        },
        "runs": runs,
        "mean_full_development_delta": float(np.mean([
            run["full_development_delta"] for run in runs
        ])),
        "min_full_development_delta": float(np.min([
            run["full_development_delta"] for run in runs
        ])),
        "max_full_development_delta": float(np.max([
            run["full_development_delta"] for run in runs
        ])),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
