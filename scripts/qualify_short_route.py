#!/usr/bin/env python3
"""Qualify the locked architecture-driven short-route candidate without final rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


SEGMENTS = (("0-16", 0, 16), ("17-32", 17, 32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path, default=Path("build-local/estimate"))
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def distances(frame: pd.DataFrame) -> np.ndarray:
    source = frame["From"].str.extract(r"^SRB_(\d+)_(\d+)/").astype(np.int16)
    target = frame["To"].str.extract(r"^SRB_(\d+)_(\d+)/").astype(np.int16)
    return ((source[0] - target[0]).abs() + (source[1] - target[1]).abs()).to_numpy(np.int16)


def row_scores(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    scores = np.full(len(truth), 100.0)
    selected = truth > 0
    relative = np.abs(prediction[selected] - truth[selected]) / truth[selected]
    scores[selected] = (1.0 - np.tanh(4.0 * relative)) * 100.0
    return scores


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    selected = truth > 0
    relative = np.abs(prediction[selected] - truth[selected]) / truth[selected]
    return {
        "rows": int(selected.sum()),
        "official_score": float(np.mean(1.0 - np.tanh(4.0 * relative)) * 100.0),
        "mean_relative_error": float(np.mean(relative)),
        "within_5_percent": float(np.mean(relative <= 0.05) * 100.0),
        "within_10_percent": float(np.mean(relative <= 0.10) * 100.0),
    }


def run_estimate(command: list[str]) -> dict[str, float | str]:
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    return {"wall_seconds": time.perf_counter() - start, "stdout": completed.stdout.strip()}


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
    short = raw.loc[raw["manhattan"] <= 32].reset_index(drop=True)
    request = args.output_dir / "request.csv"
    baseline_path = args.output_dir / "baseline.csv"
    candidate_path = args.output_dir / "candidate.csv"
    short[["From", "To"]].to_csv(request, index=False)
    common = [str(args.estimate), "-in", str(request), "--arch", str(args.arch)]
    baseline_run = run_estimate(common + ["-out", str(baseline_path)])
    candidate_run = run_estimate(common + ["-out", str(candidate_path), "--short-route"])
    truth = short["delay"].to_numpy(np.float64)
    baseline = pd.read_csv(baseline_path, usecols=["delay"])["delay"].to_numpy(np.float64)
    candidate = pd.read_csv(candidate_path, usecols=["delay"])["delay"].to_numpy(np.float64)
    row_id = short["row_id"].to_numpy(np.int32)
    distance = short["manhattan"].to_numpy(np.int16)
    baseline_scores = row_scores(truth, baseline)
    candidate_scores = row_scores(truth, candidate)
    report: dict[str, object] = {
        "protocol": {
            "architecture_only_search": True,
            "calibration_training_rows": "0-799999",
            "development_rows": "800000-899999",
            "final_rows_read": False,
            "sealed_labels_read": False,
            "locked_cli": "--short-route",
            "heuristic_weight_milli": 1200,
            "segments": {
                "0-16": {"multiplier_ppm": 999000},
                "17-32": {"multiplier_ppm": 997750},
            },
        },
        "runs": {"baseline": baseline_run, "candidate": candidate_run},
        "splits": {},
        "development_25k_blocks": {},
    }
    for split_name, selected, denominator in (
        ("training", row_id < 800_000, 800_000),
        ("development", row_id >= 800_000, 100_000),
    ):
        split: dict[str, object] = {
            "full_score_delta": float(
                np.sum(candidate_scores[selected] - baseline_scores[selected]) / denominator
            ),
            "segments": {},
        }
        for name, low, high in SEGMENTS:
            segment = selected & (distance >= low) & (distance <= high)
            split["segments"][name] = {
                "baseline": metrics(truth[segment], baseline[segment]),
                "candidate": metrics(truth[segment], candidate[segment]),
            }
        report["splits"][split_name] = split
    for low in (800_000, 825_000, 850_000, 875_000):
        block: dict[str, object] = {}
        for name, segment_low, segment_high in SEGMENTS:
            selected = (
                (row_id >= low) & (row_id < low + 25_000) &
                (distance >= segment_low) & (distance <= segment_high)
            )
            block[name] = metrics(truth[selected], candidate[selected])
        report["development_25k_blocks"][f"{low}-{low + 24_999}"] = block
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
