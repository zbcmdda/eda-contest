#!/usr/bin/env python3
"""Generate deterministic, architecture-focused unlabeled endpoint pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--profile", choices=["mixed", "short"], default="mixed")
    parser.add_argument("--template-rows", type=int, default=900_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_endpoint(text: str) -> tuple[int, int, str]:
    site, port = text.split("/", 1)
    _, x, y = site.split("_")
    return int(x), int(y), port


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("candidate count must be positive")
    all_raw = pd.read_csv(args.answers, dtype={"From": "string", "To": "string"}, usecols=["From", "To"])
    if args.template_rows <= 0 or args.template_rows > len(all_raw):
        raise ValueError("template row count must be positive and no larger than the answer set")
    raw = all_raw.iloc[: args.template_rows].reset_index(drop=True)
    sources = raw["From"].drop_duplicates().tolist()
    targets = raw["To"].drop_duplicates().tolist()
    known = set(zip(all_raw["From"], all_raw["To"]))
    source_info = {value: parse_endpoint(value) for value in sources}
    target_info = {value: parse_endpoint(value) for value in targets}
    sources_by_port: dict[str, list[str]] = defaultdict(list)
    for value, (_, _, port) in source_info.items():
        sources_by_port[port].append(value)
    raw_source = raw["From"].map(parse_endpoint)
    raw_target = raw["To"].map(parse_endpoint)
    raw_dx = np.fromiter((target[0] - source[0] for source, target in zip(raw_source, raw_target)), dtype=np.int16)
    raw_dy = np.fromiter((target[1] - source[1] for source, target in zip(raw_source, raw_target)), dtype=np.int16)
    raw_distance = np.abs(raw_dx) + np.abs(raw_dy)
    short_empirical: dict[str, list[tuple[str, int, int, str]]] = {}
    for stratum, low, high in (("short_0_16", 0, 16), ("short_17_32", 17, 32)):
        mask = (raw_distance >= low) & (raw_distance <= high)
        indices = np.flatnonzero(mask)
        short_empirical[stratum] = [
            (
                raw_source.iloc[index][2],
                int(raw_dx[index]),
                int(raw_dy[index]),
                raw_target.iloc[index][2],
            )
            for index in indices
        ]
    with (args.arch / "SRB_Inst.json").open(encoding="utf-8") as stream:
        active_sites = {(int(value["x"]), int(value["y"])) for value in json.load(stream)["Inst"]}
    targets_by_site: dict[tuple[int, int], list[str]] = defaultdict(list)
    for value, (x, y, _) in target_info.items():
        targets_by_site[(x, y)].append(value)
    source_port_counts = Counter(port for _, _, port in source_info.values())
    target_port_counts = Counter(port for _, _, port in target_info.values())
    rare_sources = [value for value in sources if source_port_counts[source_info[value][2]] <= 2_000]
    rare_targets = [value for value in targets if target_port_counts[target_info[value][2]] <= 4_000]

    with (args.arch / "SRB_Gap.json").open(encoding="utf-8") as stream:
        blocks = json.load(stream)["Gap"]["Block"]
    block_sources: list[str] = []
    block_targets: list[str] = []
    for value, (x, y, _) in source_info.items():
        if any(block["lower"] <= y <= block["upper"] and (x < block["left"] or x > block["right"])
               for block in blocks):
            block_sources.append(value)
    for value, (x, y, _) in target_info.items():
        if any(block["lower"] <= y <= block["upper"] and (x < block["left"] or x > block["right"])
               for block in blocks):
            block_targets.append(value)

    rng = np.random.default_rng(args.seed)
    selected: list[tuple[str, str, str]] = []
    selected_pairs: set[tuple[str, str]] = set()
    if args.profile == "short":
        quotas = {
            "short_0_16": round(args.count * 0.55),
            "short_17_32": args.count - round(args.count * 0.55),
            "block_band": 0,
            "rare_port": 0,
            "public_like": 0,
        }
    else:
        quotas = {
            "short_0_16": round(args.count * 0.35),
            "short_17_32": round(args.count * 0.20),
            "block_band": round(args.count * 0.15),
            "rare_port": round(args.count * 0.15),
        }
        quotas["public_like"] = args.count - sum(quotas.values())
    selected_counts = Counter()

    def add(source: str, target: str, stratum: str) -> bool:
        pair = (source, target)
        if pair in known or pair in selected_pairs:
            return False
        selected_pairs.add(pair)
        selected.append((source, target, stratum))
        selected_counts[stratum] += 1
        return True

    def add_short(stratum: str, low: int, high: int, quota: int) -> None:
        attempts = 0
        while selected_counts[stratum] < quota:
            attempts += 1
            if attempts > quota * 500:
                raise RuntimeError(f"could not fill {stratum} quota")
            if args.profile == "short":
                template_pool = short_empirical[stratum]
                source_port, dx, dy, target_port = template_pool[
                    int(rng.integers(len(template_pool)))
                ]
                source_pool = sources_by_port[source_port]
                source = source_pool[int(rng.integers(len(source_pool)))]
            else:
                source = sources[int(rng.integers(len(sources)))]
                dx = int(rng.integers(-high, high + 1))
                dy_limit = high - abs(dx)
                if dy_limit < 0:
                    continue
                dy = int(rng.integers(-dy_limit, dy_limit + 1))
                target_port = ""
            sx, sy, _ = source_info[source]
            distance = abs(dx) + abs(dy)
            if distance < low or distance > high:
                continue
            site = (sx + dx, sy + dy)
            if args.profile == "short":
                if site in active_sites:
                    add(source, f"SRB_{site[0]}_{site[1]}/{target_port}", stratum)
            else:
                choices = targets_by_site.get(site)
                if choices:
                    add(source, choices[int(rng.integers(len(choices)))], stratum)

    add_short("short_0_16", 0, 16, quotas["short_0_16"])
    add_short("short_17_32", 17, 32, quotas["short_17_32"])

    for stratum, source_pool, target_pool in (
        ("block_band", block_sources, block_targets),
        ("rare_port", rare_sources, rare_targets),
        ("public_like", sources, targets),
    ):
        quota = quotas[stratum]
        if quota == 0:
            continue
        before = len(selected)
        attempts = 0
        while len(selected) - before < quota:
            attempts += 1
            if attempts > quota * 200:
                raise RuntimeError(f"could not fill {stratum} quota")
            source = source_pool[int(rng.integers(len(source_pool)))]
            target = target_pool[int(rng.integers(len(target_pool)))]
            if stratum == "block_band":
                sx, sy, _ = source_info[source]
                tx, ty, _ = target_info[target]
                crosses = any(
                    block["lower"] <= sy <= block["upper"]
                    and block["lower"] <= ty <= block["upper"]
                    and ((sx < block["left"] and tx > block["right"])
                         or (sx > block["right"] and tx < block["left"]))
                    for block in blocks
                )
                if not crosses:
                    continue
            add(source, target, stratum)

    frame = pd.DataFrame(selected, columns=["From", "To", "stratum"])
    if len(frame) != args.count:
        raise RuntimeError(f"generated {len(frame)} candidates, expected {args.count}")
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[["From", "To"]].to_csv(args.output, index=False)
    metadata = {
        "count": len(frame),
        "seed": args.seed,
        "template_rows": args.template_rows,
        "strata": frame["stratum"].value_counts().sort_index().to_dict(),
        "known_pairs_excluded": len(known),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frame[["From", "To", "stratum"]].to_csv(args.output.with_suffix(".metadata.csv"), index=False)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
