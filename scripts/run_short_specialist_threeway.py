#!/usr/bin/env python3
"""Select a short-path specialist on development data, then evaluate once on test."""

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
    parser.add_argument("--extra-answers", type=Path, required=True)
    parser.add_argument("--base-development", type=Path, required=True)
    parser.add_argument("--base-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-rows", type=int, default=800_000)
    parser.add_argument("--development-rows", type=int, default=100_000)
    parser.add_argument("--threshold", type=int, default=32)
    parser.add_argument("--extra-weights", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def read_aligned_predictions(path: Path, expected: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"From": "string", "To": "string", "delay": np.int32})
    if len(frame) != len(expected) or not frame[["From", "To"]].reset_index(drop=True).equals(
        expected[["From", "To"]].reset_index(drop=True)
    ):
        raise ValueError(f"prediction file does not align: {path}")
    return frame["delay"].to_numpy(dtype=np.float64)


def main() -> None:
    args = parse_args()
    public = pd.read_csv(args.answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    extra = pd.read_csv(args.extra_answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    development_end = args.training_rows + args.development_rows
    if args.training_rows <= 0 or development_end >= len(public):
        raise ValueError("three-way split must leave a non-empty final test set")
    development_raw = public.iloc[args.training_rows:development_end]
    test_raw = public.iloc[development_end:]
    base_development = read_aligned_predictions(args.base_development, development_raw)
    base_test = read_aligned_predictions(args.base_test, test_raw)

    architecture = ArchitectureFeatures(args.arch)
    combined = pd.concat([public, extra], ignore_index=True)
    combined_features = architecture.build(combined)
    public_features = combined_features.iloc[: len(public)]
    extra_features = combined_features.iloc[len(public) :]
    categorical = [
        column for column in combined_features
        if isinstance(combined_features[column].dtype, pd.CategoricalDtype)
    ]
    train_x = public_features.iloc[: args.training_rows]
    development_x = public_features.iloc[args.training_rows:development_end]
    test_x = public_features.iloc[development_end:]
    train_y = public["delay"].to_numpy(dtype=np.float64)[: args.training_rows]
    development_y = development_raw["delay"].to_numpy(dtype=np.float64)
    test_y = test_raw["delay"].to_numpy(dtype=np.float64)
    train_mask = train_x["manhattan"].to_numpy() <= args.threshold
    development_mask = development_x["manhattan"].to_numpy() <= args.threshold
    test_mask = test_x["manhattan"].to_numpy() <= args.threshold
    extra_mask = extra_features["manhattan"].to_numpy() <= args.threshold

    selected_x = pd.concat(
        [train_x.loc[train_mask], extra_features.loc[extra_mask]], ignore_index=True
    )
    selected_raw_y = np.concatenate(
        [train_y[train_mask], extra["delay"].to_numpy(dtype=np.float64)[extra_mask]]
    )
    selected_y = np.log1p(selected_raw_y)
    public_selected_rows = int(train_mask.sum())
    extra_selected_rows = int(extra_mask.sum())
    parameters = {
        "objective": "regression",
        "metric": "l1",
        "num_leaves": 127,
        "max_depth": 8,
        "learning_rate": 0.08,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.95,
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
    models: dict[float, lgb.Booster] = {}
    for extra_weight in args.extra_weights:
        weights = np.concatenate(
            [
                np.ones(public_selected_rows, dtype=np.float64),
                np.full(extra_selected_rows, extra_weight, dtype=np.float64),
            ]
        )
        dataset = lgb.Dataset(
            selected_x,
            label=selected_y,
            weight=weights,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        model = lgb.train(parameters, dataset, num_boost_round=128)
        models[extra_weight] = model
        specialist = np.maximum(
            0.0, np.expm1(model.predict(development_x.loc[development_mask]))
        )
        hybrid = base_development.copy()
        hybrid[development_mask] = np.rint(specialist)
        hybrid[development_y == 0] = 0
        run = {
            "extra_weight": extra_weight,
            "hybrid_metrics": official_metrics(development_y, hybrid),
            "specialist_metrics": official_metrics(
                development_y[development_mask], np.rint(specialist)
            ),
        }
        development_runs.append(run)
        print(json.dumps({"split": "development", **run}, indent=2, sort_keys=True), flush=True)

    selected = max(
        development_runs, key=lambda value: value["hybrid_metrics"]["official_score"]
    )
    selected_weight = float(selected["extra_weight"])
    selected_model = models[selected_weight]
    test_specialist = np.maximum(0.0, np.expm1(selected_model.predict(test_x.loc[test_mask])))
    test_hybrid = base_test.copy()
    test_hybrid[test_mask] = np.rint(test_specialist)
    test_hybrid[test_y == 0] = 0
    selected_model_path = args.output_dir / "selected_short_model.txt"
    selected_model.save_model(selected_model_path)
    report = {
        "protocol": {
            "training_rows": args.training_rows,
            "development_rows": args.development_rows,
            "test_rows": len(test_raw),
            "threshold": args.threshold,
            "public_short_training_rows": public_selected_rows,
            "extra_short_training_rows": extra_selected_rows,
        },
        "base_development_metrics": official_metrics(development_y, base_development),
        "base_test_metrics": official_metrics(test_y, base_test),
        "development_runs": development_runs,
        "selected_weight": selected_weight,
        "selected_model": str(selected_model_path),
        "test_hybrid_metrics": official_metrics(test_y, test_hybrid),
        "test_specialist_metrics": official_metrics(
            test_y[test_mask], np.rint(test_specialist)
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"split": "final_test", **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
