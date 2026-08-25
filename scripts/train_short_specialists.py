#!/usr/bin/env python3
"""Train short-distance specialists and evaluate deterministic hybrid routing."""

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
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--extra-answers", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--validation-rows", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--variants", nargs="+", choices=["d8_t128", "d10_t256"], default=["d8_t128", "d10_t256"])
    parser.add_argument("--extra-weights", type=float, nargs="+", default=[1.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    extra = None
    if args.extra_answers is not None:
        extra = pd.read_csv(
            args.extra_answers, dtype={"From": "string", "To": "string", "delay": np.int32}
        )
    base_frame = pd.read_csv(
        args.base_predictions, dtype={"From": "string", "To": "string", "delay": np.int32}
    )
    if len(raw) != len(base_frame) or not raw[["From", "To"]].equals(base_frame[["From", "To"]]):
        raise ValueError("base predictions do not align with answers")
    split = len(raw) - args.validation_rows
    architecture = ArchitectureFeatures(args.arch)
    combined = raw if extra is None else pd.concat([raw, extra], ignore_index=True)
    combined_features = architecture.build(combined)
    features = combined_features.iloc[: len(raw)]
    extra_features = None if extra is None else combined_features.iloc[len(raw) :]
    categorical = [column for column in features if isinstance(features[column].dtype, pd.CategoricalDtype)]
    train_x = features.iloc[:split]
    valid_x = features.iloc[split:]
    train_y = raw["delay"].to_numpy(dtype=np.float64)[:split]
    valid_y = raw["delay"].to_numpy(dtype=np.float64)[split:]
    base = base_frame["delay"].to_numpy(dtype=np.float64)[split:]
    baseline_metrics = official_metrics(valid_y, base)
    variants = [
        {"name": "d8_t128", "trees": 128, "depth": 8, "leaves": 127, "rate": 0.08},
        {"name": "d10_t256", "trees": 256, "depth": 10, "leaves": 255, "rate": 0.06},
    ]
    variants = [variant for variant in variants if variant["name"] in args.variants]
    results: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for threshold in args.thresholds:
        train_mask = train_x["manhattan"].to_numpy() <= threshold
        valid_mask = valid_x["manhattan"].to_numpy() <= threshold
        extra_mask = None if extra_features is None else extra_features["manhattan"].to_numpy() <= threshold
        for variant in variants:
          for extra_weight in args.extra_weights:
            selected_x = train_x.loc[train_mask]
            selected_raw_y = train_y[train_mask]
            selected_weights = np.ones(len(selected_x), dtype=np.float64)
            if extra_features is not None and extra is not None:
                selected_x = pd.concat([selected_x, extra_features.loc[extra_mask]], ignore_index=True)
                selected_raw_y = np.concatenate(
                    [selected_raw_y, extra["delay"].to_numpy(dtype=np.float64)[extra_mask]]
                )
                selected_weights = np.concatenate(
                    [selected_weights, np.full(int(extra_mask.sum()), extra_weight, dtype=np.float64)]
                )
            selected_y = np.log1p(selected_raw_y)
            dataset = lgb.Dataset(
                selected_x,
                label=selected_y,
                weight=selected_weights,
                categorical_feature=categorical,
                free_raw_data=False,
            )
            parameters = {
                "objective": "regression",
                "metric": "l1",
                "num_leaves": variant["leaves"],
                "max_depth": variant["depth"],
                "learning_rate": variant["rate"],
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
            model = lgb.train(parameters, dataset, num_boost_round=variant["trees"])
            prediction = np.maximum(0.0, np.expm1(model.predict(valid_x.loc[valid_mask])))
            hybrid = base.copy()
            hybrid[valid_mask] = np.rint(prediction)
            hybrid[valid_y == 0] = 0
            weight_name = str(extra_weight).replace(".", "p")
            stem = f"manhattan_{threshold}_{variant['name']}_w{weight_name}"
            model_path = args.output_dir / f"{stem}.txt"
            model.save_model(model_path)
            run = {
                "threshold": threshold,
                "variant": variant["name"],
                "training_rows": int(train_mask.sum()),
                "extra_training_rows": 0 if extra_mask is None else int(extra_mask.sum()),
                "extra_weight": extra_weight,
                "validation_rows": int(valid_mask.sum()),
                "specialist_metrics": official_metrics(valid_y[valid_mask], np.rint(prediction)),
                "base_subset_metrics": official_metrics(valid_y[valid_mask], base[valid_mask]),
                "hybrid_metrics": official_metrics(valid_y, hybrid),
                "model": str(model_path),
            }
            results.append(run)
            (args.output_dir / f"{stem}.json").write_text(
                json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(run, indent=2, sort_keys=True), flush=True)
    report = {"baseline_metrics": baseline_metrics, "runs": results}
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
