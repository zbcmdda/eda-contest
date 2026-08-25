#!/usr/bin/env python3
"""Multi-seed/objective robustness check for the two-hop short residual model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from short_features import TWO_HOP_NAMES
from train_model import ArchitectureFeatures, official_metrics


SEGMENTS = (("0-16", 0, 16), ("17-32", 17, 32), ("33-64", 33, 64), ("65-128", 65, 128))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def endpoint_distance(frame: pd.DataFrame) -> np.ndarray:
    source = frame["From"].str.extract(r"^SRB_(\d+)_(\d+)/").astype(np.int16)
    target = frame["To"].str.extract(r"^SRB_(\d+)_(\d+)/").astype(np.int16)
    return ((source[0] - target[0]).abs() + (source[1] - target[1]).abs()).to_numpy(np.int16)


def row_scores(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full(len(truth), 100.0, dtype=np.float64)
    positive = truth > 0
    relative = np.abs(prediction[positive] - truth[positive]) / truth[positive]
    result[positive] = (1.0 - np.tanh(4.0 * relative)) * 100.0
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
    raw["manhattan"] = endpoint_distance(raw)
    short = raw.loc[raw["manhattan"] <= 128].reset_index(drop=True)
    train = short["row_id"].to_numpy() < 800_000
    development = ~train

    base = ArchitectureFeatures(args.arch).build(short[["From", "To", "delay"]])
    local = pd.read_csv(
        args.features_dir / "short_0_128.features.csv", usecols=TWO_HOP_NAMES, dtype=np.int32
    )
    features = pd.concat([base.reset_index(drop=True), local.reset_index(drop=True)], axis=1)
    categorical = [
        column for column in base if isinstance(base[column].dtype, pd.CategoricalDtype)
    ]
    baseline = pd.read_csv(
        args.features_dir / "short_0_128.baseline.csv", usecols=["delay"]
    )["delay"].to_numpy(dtype=np.float64)
    truth = short["delay"].to_numpy(dtype=np.float64)
    distance = short["manhattan"].to_numpy(dtype=np.int16)
    row_id = short["row_id"].to_numpy(dtype=np.int32)
    train_label = np.log((truth[train] + 1.0) / (baseline[train] + 1.0))

    parameter_base = {
        "metric": "l1",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": args.threads,
        "force_col_wise": True,
    }
    runs = []
    seeds = (20260815, 20260816, 20260817, 20260824, 20260825)
    configurations = [("regression", seed) for seed in seeds]
    configurations += [("regression_l1", seed) for seed in seeds]
    configurations += [(objective, 20260824) for objective in ("huber", "fair")]
    development_truth = truth[development]
    development_baseline = baseline[development]
    development_distance = distance[development]
    development_rows = row_id[development]
    base_row_scores = row_scores(development_truth, development_baseline)

    for objective, seed in configurations:
        model = lgb.train(
            {**parameter_base, "objective": objective, "seed": seed},
            lgb.Dataset(
                features.loc[train],
                label=train_label,
                categorical_feature=categorical,
                free_raw_data=False,
            ),
            num_boost_round=64,
        )
        residual = model.predict(features.loc[development])
        model_name = f"residual_A2_{objective}_seed{seed}_64"
        model.save_model(str(args.output_dir / f"{model_name}.txt"))
        for alpha in (1.0, 1.25, 1.5):
            prediction = np.rint(
                np.maximum(0.0, (development_baseline + 1.0) * np.exp(alpha * residual) - 1.0)
            )
            prediction[development_baseline == 0] = 0
            score = official_metrics(development_truth, prediction)
            predicted_row_scores = row_scores(development_truth, prediction)
            report = {
                "model": model_name,
                "objective": objective,
                "seed": seed,
                "alpha": alpha,
                "short_score": score["official_score"],
                "short_delta": score["official_score"] - official_metrics(
                    development_truth, development_baseline
                )["official_score"],
                "full_development_delta": float(
                    np.sum(predicted_row_scores - base_row_scores) / 100_000.0
                ),
                "segments": {},
                "contiguous_blocks": {},
            }
            for name, low, high in SEGMENTS:
                selected = (development_distance >= low) & (development_distance <= high)
                candidate_score = official_metrics(development_truth[selected], prediction[selected])[
                    "official_score"
                ]
                baseline_score = official_metrics(
                    development_truth[selected], development_baseline[selected]
                )["official_score"]
                report["segments"][name] = candidate_score - baseline_score
            for low in (800_000, 825_000, 850_000, 875_000):
                selected = (development_rows >= low) & (development_rows < low + 25_000)
                report["contiguous_blocks"][f"{low}-{low + 24_999}"] = float(
                    np.sum(predicted_row_scores[selected] - base_row_scores[selected]) / 25_000.0
                )
            runs.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)

    natural = [run for run in runs if run["objective"] == "regression"]
    summary = {
        "protocol": {
            "training": "public rows 0-799999, Manhattan <= 128",
            "development": "public rows 800000-899999",
            "frozen_rows_read": False,
            "sealed_labels_read": False,
        },
        "runs": runs,
        "regression_seed_summary": {
            str(alpha): {
                "mean_full_delta": float(np.mean([
                    run["full_development_delta"] for run in natural if run["alpha"] == alpha
                ])),
                "min_full_delta": float(np.min([
                    run["full_development_delta"] for run in natural if run["alpha"] == alpha
                ])),
                "max_full_delta": float(np.max([
                    run["full_development_delta"] for run in natural if run["alpha"] == alpha
                ])),
            }
            for alpha in (1.0, 1.25, 1.5)
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
