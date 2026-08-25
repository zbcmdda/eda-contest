#!/usr/bin/env python3
"""Search low-cost 0--128 path specialists on the public development split.

Rows 0--799,999 are the only labeled training rows. Rows 800,000--899,999 are
the development block. The frozen final 100,000 rows and sealed pilot labels
are deliberately not read by this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from short_features import NAMES, TWO_HOP_NAMES
from train_model import ArchitectureFeatures, official_metrics


SEGMENTS = (("0-16", 0, 16), ("17-32", 17, 32), ("33-64", 33, 64), ("65-128", 65, 128))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path, default=Path("build/estimate"))
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def manhattan(frame: pd.DataFrame) -> np.ndarray:
    source = frame["From"].str.extract(r"^SRB_(\d+)_(\d+)/")
    target = frame["To"].str.extract(r"^SRB_(\d+)_(\d+)/")
    return (
        (source[0].astype(np.int16) - target[0].astype(np.int16)).abs()
        + (source[1].astype(np.int16) - target[1].astype(np.int16)).abs()
    ).to_numpy(dtype=np.int16)


def run_csv(executable: Path, arch: Path, request: Path, output: Path, extra: list[str]) -> None:
    command = [str(executable), *extra, "-in", str(request), "-out", str(output), "--arch", str(arch)]
    subprocess.run(command, check=True)


def metrics_by_segment(
    truth: np.ndarray,
    prediction: np.ndarray,
    distances: np.ndarray,
) -> dict[str, dict[str, float]]:
    result = {"0-128": official_metrics(truth, prediction)}
    for name, low, high in SEGMENTS:
        selected = (distances >= low) & (distances <= high)
        result[name] = {"rows": int(selected.sum()), **official_metrics(truth[selected], prediction[selected])}
    return result


def prediction_report(
    name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    distances: np.ndarray,
    full_truth: np.ndarray,
    full_baseline: np.ndarray,
    full_short_mask: np.ndarray,
) -> dict:
    prediction = np.rint(np.maximum(0.0, prediction)).astype(np.float64)
    prediction[baseline == 0] = 0
    full_prediction = full_baseline.copy()
    full_prediction[full_short_mask] = prediction
    grouped = metrics_by_segment(truth, prediction, distances)
    baseline_grouped = metrics_by_segment(truth, baseline, distances)
    return {
        "name": name,
        "short": grouped,
        "short_delta": {
            key: grouped[key]["official_score"] - baseline_grouped[key]["official_score"]
            for key in grouped
        },
        "full_development": official_metrics(full_truth, full_prediction),
        "full_development_delta": (
            official_metrics(full_truth, full_prediction)["official_score"]
            - official_metrics(full_truth, full_baseline)["official_score"]
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        args.answers,
        nrows=900_000,
        dtype={"From": "string", "To": "string", "delay": np.int32},
    )
    raw["row_id"] = np.arange(len(raw), dtype=np.int32)
    raw["manhattan"] = manhattan(raw)
    short = raw.loc[raw["manhattan"] <= 128].reset_index(drop=True)
    train_mask = short["row_id"].to_numpy() < 800_000
    dev_mask = ~train_mask

    request_path = args.output_dir / "short_0_128.request.csv"
    baseline_path = args.output_dir / "short_0_128.baseline.csv"
    short_feature_path = args.output_dir / "short_0_128.features.csv"
    short[["From", "To"]].to_csv(request_path, index=False)
    run_csv(args.estimate, args.arch, request_path, baseline_path, [])
    run_csv(
        args.estimate,
        args.arch,
        request_path,
        short_feature_path,
        ["--dump-short-features", "--limit", str(len(short))],
    )

    baseline = pd.read_csv(baseline_path, usecols=["delay"])["delay"].to_numpy(dtype=np.float64)
    local = pd.read_csv(short_feature_path, usecols=NAMES, dtype=np.int32)
    base = ArchitectureFeatures(args.arch).build(short[["From", "To", "delay"]])
    features = pd.concat([base.reset_index(drop=True), local.reset_index(drop=True)], axis=1)
    categorical = [
        column for column in base if isinstance(base[column].dtype, pd.CategoricalDtype)
    ]

    train_y = short.loc[train_mask, "delay"].to_numpy(dtype=np.float64)
    dev_y = short.loc[dev_mask, "delay"].to_numpy(dtype=np.float64)
    dev_baseline = baseline[dev_mask]
    dev_distance = short.loc[dev_mask, "manhattan"].to_numpy(dtype=np.int16)

    full_dev = raw.loc[raw["row_id"] >= 800_000].reset_index(drop=True)
    full_request = args.output_dir / "development.request.csv"
    full_baseline_path = args.output_dir / "development.baseline.csv"
    full_dev[["From", "To"]].to_csv(full_request, index=False)
    run_csv(args.estimate, args.arch, full_request, full_baseline_path, [])
    full_baseline = pd.read_csv(full_baseline_path, usecols=["delay"])["delay"].to_numpy(dtype=np.float64)
    full_truth = full_dev["delay"].to_numpy(dtype=np.float64)
    full_short_mask = full_dev["manhattan"].to_numpy(dtype=np.int16) <= 128
    if not np.array_equal(full_baseline[full_short_mask], dev_baseline):
        raise ValueError("short and full development baselines do not align")

    common = {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.06,
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
    variants = (
        ("A0", list(base.columns)),
        ("A1", list(base.columns) + NAMES[:17]),
        ("A2", list(base.columns) + TWO_HOP_NAMES),
    )
    reports: list[dict] = []
    baseline_report = prediction_report(
        "production_baseline",
        dev_y,
        dev_baseline,
        dev_baseline,
        dev_distance,
        full_truth,
        full_baseline,
        full_short_mask,
    )
    reports.append(baseline_report)
    print(json.dumps(baseline_report, sort_keys=True), flush=True)

    for variant, columns in variants:
        train_x = features.loc[train_mask, columns]
        dev_x = features.loc[dev_mask, columns]
        cats = [column for column in columns if column in categorical]
        for balanced in (False, True):
            weights = None
            suffix = "natural"
            if balanced:
                train_distance = short.loc[train_mask, "manhattan"].to_numpy(dtype=np.int16)
                weights = np.ones(len(train_distance), dtype=np.float64)
                counts = []
                for _, low, high in SEGMENTS:
                    selected = (train_distance >= low) & (train_distance <= high)
                    counts.append(int(selected.sum()))
                target_count = sum(counts) / len(counts)
                for count, (_, low, high) in zip(counts, SEGMENTS):
                    selected = (train_distance >= low) & (train_distance <= high)
                    weights[selected] = target_count / count
                suffix = "balanced"

            direct = lgb.train(
                {**common, "num_leaves": 63, "max_depth": 8},
                lgb.Dataset(
                    train_x,
                    label=np.log1p(train_y),
                    weight=weights,
                    categorical_feature=cats,
                    free_raw_data=False,
                ),
                num_boost_round=96,
            )
            direct_prediction = np.expm1(direct.predict(dev_x))
            name = f"direct_{variant}_{suffix}_96"
            direct.save_model(str(args.output_dir / f"{name}.txt"))
            report = prediction_report(
                name,
                dev_y,
                direct_prediction,
                dev_baseline,
                dev_distance,
                full_truth,
                full_baseline,
                full_short_mask,
            )
            reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)

            for trees in (32, 48, 64):
                residual = lgb.train(
                    {**common, "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05},
                    lgb.Dataset(
                        train_x,
                        label=np.log((train_y + 1.0) / (baseline[train_mask] + 1.0)),
                        weight=weights,
                        categorical_feature=cats,
                        free_raw_data=False,
                    ),
                    num_boost_round=trees,
                )
                residual_prediction = (
                    (dev_baseline + 1.0) * np.exp(residual.predict(dev_x)) - 1.0
                )
                name = f"residual_{variant}_{suffix}_{trees}"
                residual.save_model(str(args.output_dir / f"{name}.txt"))
                report = prediction_report(
                    name,
                    dev_y,
                    residual_prediction,
                    dev_baseline,
                    dev_distance,
                    full_truth,
                    full_baseline,
                    full_short_mask,
                )
                reports.append(report)
                print(json.dumps(report, sort_keys=True), flush=True)

    best = max(reports[1:], key=lambda item: item["full_development_delta"])
    output = {
        "protocol": {
            "training_rows": "public rows 0-799999, Manhattan <= 128",
            "development_rows": "public rows 800000-899999",
            "frozen_rows_read": False,
            "sealed_labels_read": False,
            "seed": args.seed,
            "short_training_rows": int(train_mask.sum()),
            "short_development_rows": int(dev_mask.sum()),
        },
        "baseline": baseline_report,
        "best": best,
        "runs": reports,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("BEST", json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
