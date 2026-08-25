#!/usr/bin/env python3
"""Select training weights on development data and evaluate one model on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_model import ArchitectureFeatures, official_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-rows", type=int, default=800_000)
    parser.add_argument("--development-rows", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def make_weights(name: str, features: pd.DataFrame, delay: np.ndarray) -> np.ndarray:
    weights = np.ones(len(features), dtype=np.float64)
    if name == "uniform":
        return weights
    kind, threshold, factor = name.split("_")
    threshold_value = float(threshold)
    factor_value = float(factor[1:])
    if kind == "distance":
        selected = features["manhattan"].to_numpy() <= threshold_value
    elif kind == "delay":
        selected = delay <= threshold_value
    else:
        raise ValueError(f"unknown weight kind: {kind}")
    weights[selected] = factor_value
    return weights


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    development_end = args.training_rows + args.development_rows
    if args.training_rows <= 0 or development_end >= len(raw):
        raise ValueError("three-way split must leave a non-empty final test set")
    architecture = ArchitectureFeatures(args.arch)
    features = architecture.build(raw)
    categorical = [column for column in features if isinstance(features[column].dtype, pd.CategoricalDtype)]
    train_x = features.iloc[: args.training_rows]
    development_x = features.iloc[args.training_rows:development_end]
    test_x = features.iloc[development_end:]
    labels = raw["delay"].to_numpy(dtype=np.float64)
    train_raw_y = labels[: args.training_rows]
    train_y = np.log1p(train_raw_y)
    development_y = labels[args.training_rows:development_end]
    test_y = labels[development_end:]
    variants = [
        "uniform",
        "distance_16_x2",
        "distance_16_x4",
        "distance_32_x2",
        "distance_32_x4",
        "distance_32_x8",
        "distance_64_x2",
        "distance_64_x4",
        "delay_750_x2",
        "delay_750_x4",
        "delay_1500_x2",
        "delay_1500_x4",
    ]
    parameters = {
        "objective": "regression",
        "metric": "l1",
        "num_leaves": 127,
        "max_depth": 8,
        "learning_rate": 0.08,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": args.threads,
        "force_col_wise": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_runs: list[dict[str, object]] = []
    models: dict[str, lgb.Booster] = {}
    for variant in variants:
        weights = make_weights(variant, train_x, train_raw_y)
        dataset = lgb.Dataset(
            train_x,
            label=train_y,
            weight=weights,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        model = lgb.train(parameters, dataset, num_boost_round=128)
        models[variant] = model
        prediction = np.maximum(0.0, np.expm1(model.predict(development_x)))
        prediction[development_y == 0] = 0
        run = {
            "variant": variant,
            "weighted_rows": int(np.count_nonzero(weights != 1.0)),
            "metrics": official_metrics(development_y, np.rint(prediction)),
        }
        development_runs.append(run)
        print(json.dumps({"split": "development", **run}, indent=2, sort_keys=True), flush=True)
    selected = max(development_runs, key=lambda value: value["metrics"]["official_score"])
    selected_name = str(selected["variant"])
    selected_model = models[selected_name]
    test_prediction = np.maximum(0.0, np.expm1(selected_model.predict(test_x)))
    test_prediction[test_y == 0] = 0
    model_path = args.output_dir / "selected_model.txt"
    selected_model.save_model(model_path)
    report = {
        "protocol": {
            "training_rows": args.training_rows,
            "development_rows": args.development_rows,
            "test_rows": len(test_x),
        },
        "development_runs": development_runs,
        "selected_variant": selected_name,
        "selected_model": str(model_path),
        "test_metrics": official_metrics(test_y, np.rint(test_prediction)),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"split": "final_test", **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
