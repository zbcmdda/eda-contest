#!/usr/bin/env python3
"""Run fixed-split LightGBM learning curves without rebuilding features per run."""

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
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 250_000, 500_000, 800_000])
    parser.add_argument("--objectives", nargs="+", choices=["log_l2", "residual_log_l2"], default=["log_l2"])
    parser.add_argument("--development-rows", type=int, default=100_000)
    parser.add_argument("--test-rows", type=int, default=100_000)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--leaves", type=int, default=127)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--topology-features",
        action="store_true",
        help="add offline local Block/boundary clearance features",
    )
    parser.add_argument(
        "--lookahead-features",
        action="store_true",
        help="add offline exact-first-step route lookahead features",
    )
    return parser.parse_args()


def transform_target(objective: str, labels: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    if objective == "residual_log_l2":
        baseline = features["relaxed_after_source"].to_numpy(dtype=np.float64)
        return np.log((labels + 1.0) / (baseline + 1.0))
    return np.log1p(labels)


def inverse_prediction(objective: str, values: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    if objective == "residual_log_l2":
        baseline = features["relaxed_after_source"].to_numpy(dtype=np.float64)
        values = (baseline + 1.0) * np.exp(values) - 1.0
    else:
        values = np.expm1(values)
    return np.maximum(0.0, values)


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    reserved = args.development_rows + args.test_rows
    if reserved <= 0 or reserved >= len(raw):
        raise ValueError("development plus test rows must reserve part of the dataset")
    train_pool_rows = len(raw) - reserved
    if any(size <= 0 or size > train_pool_rows for size in args.sizes):
        raise ValueError(f"training sizes must be between 1 and {train_pool_rows}")

    architecture = ArchitectureFeatures(args.arch)
    features = architecture.build(
        raw,
        include_topology=args.topology_features,
        include_lookahead=args.lookahead_features,
    )
    categorical = [column for column in features if isinstance(features[column].dtype, pd.CategoricalDtype)]
    development_slice = slice(train_pool_rows, train_pool_rows + args.development_rows)
    test_slice = slice(train_pool_rows + args.development_rows, len(raw))
    splits = {
        "development": (features.iloc[development_slice], raw["delay"].to_numpy(dtype=np.float64)[development_slice]),
        "test": (features.iloc[test_slice], raw["delay"].to_numpy(dtype=np.float64)[test_slice]),
    }
    parameters = {
        "objective": "regression",
        "metric": "l1",
        "num_leaves": args.leaves,
        "max_depth": args.depth,
        "learning_rate": args.learning_rate,
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
    results: list[dict[str, object]] = []
    labels = raw["delay"].to_numpy(dtype=np.float64)

    for objective in args.objectives:
        for size in sorted(set(args.sizes)):
            train_x = features.iloc[:size]
            train_raw = labels[:size]
            train_y = transform_target(objective, train_raw, train_x)
            dataset = lgb.Dataset(
                train_x,
                label=train_y,
                categorical_feature=categorical,
                free_raw_data=False,
            )
            model = lgb.train(parameters, dataset, num_boost_round=args.trees)
            stem = f"{objective}_{size}"
            model_path = args.output_dir / f"{stem}.txt"
            model.save_model(model_path)
            run: dict[str, object] = {
                "objective": objective,
                "training_rows": size,
                "model": str(model_path),
                "metrics": {},
            }
            for split_name, (split_x, split_y) in splits.items():
                predicted = inverse_prediction(objective, model.predict(split_x), split_x)
                predicted[split_y == 0] = 0
                run["metrics"][split_name] = official_metrics(split_y, predicted)
                prediction_frame = raw.iloc[
                    development_slice if split_name == "development" else test_slice
                ][["From", "To"]].copy()
                prediction_frame["delay"] = np.rint(predicted).astype(np.uint32)
                prediction_frame.to_csv(args.output_dir / f"{stem}.{split_name}.csv", index=False)
            results.append(run)
            report = {
                "configuration": {
                    "train_pool_rows": train_pool_rows,
                    "development_rows": args.development_rows,
                    "test_rows": args.test_rows,
                    "trees": args.trees,
                    "leaves": args.leaves,
                    "depth": args.depth,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                },
                "runs": results,
            }
            (args.output_dir / "results.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(run, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
