#!/usr/bin/env python3
"""Build a leakage-resistant, stratified short-route candidate pilot.

The generated files contain endpoints and design metadata only.  In particular,
the sealed split CSVs are intentionally label-free; their labels must remain
sealed until the model configuration is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


LAYERS = (("0", 0, 0), ("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16),
          ("17-24", 17, 24), ("25-32", 25, 32))
SPLITS = ("dev", "sealed_iid", "sealed_ood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--template-rows", type=int, default=800_000)
    parser.add_argument("--exclude-candidates", type=Path, action="append", default=[])
    parser.add_argument("--boundary-margin", type=int, default=4)
    parser.add_argument("--block-margin", type=int, default=3)
    parser.add_argument("--public-check-per-layer", type=int, default=100)
    return parser.parse_args()


def parse_endpoint(text: str) -> tuple[int, int, str]:
    site, port = text.split("/", 1)
    prefix, x, y = site.split("_")
    if prefix != "SRB":
        raise ValueError(f"invalid endpoint {text}")
    return int(x), int(y), port


def endpoint_key(source: str, target: str) -> str:
    return source + "\0" + target


def group_key(source_port: str, target_port: str, dx: int, dy: int) -> str:
    return json.dumps([source_port, target_port, dx, dy], separators=(",", ":"))


def stable_bucket(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "big") % 10


def layer_name(distance: int) -> str:
    for name, lower, upper in LAYERS:
        if lower <= distance <= upper:
            return name
    raise ValueError(f"distance {distance} is outside 0-32")


def candidate_files(args: argparse.Namespace) -> list[Path]:
    discovered = sorted(Path("refine-logs").glob("*candidates*.csv"))
    files = [*discovered, *args.exclude_candidates]
    return sorted({path.resolve() for path in files if path.exists() and not path.name.endswith(".metadata.csv")})


def quotas(count: int) -> dict[str, dict[str, int]]:
    if count < len(LAYERS) * len(SPLITS):
        raise ValueError("--count must allow at least one row per layer and split")
    layer_counts = {name: count // len(LAYERS) for name, _, _ in LAYERS}
    for name, _, _ in LAYERS[: count % len(LAYERS)]:
        layer_counts[name] += 1
    result = {name: {split: int(layer_counts[name] * ratio) for split, ratio in
                     zip(SPLITS, (0.6, 0.2, 0.2))} for name in layer_counts}
    targets = {"dev": count * 6 // 10, "sealed_iid": count * 2 // 10}
    targets["sealed_ood"] = count - targets["dev"] - targets["sealed_iid"]
    for name, values in result.items():
        remainder = layer_counts[name] - sum(values.values())
        # Give per-layer rounding residue to the split with the largest global deficit.
        for _ in range(remainder):
            split = max(SPLITS, key=lambda item: targets[item] - sum(v[item] for v in result.values()))
            values[split] += 1
    if sum(sum(value.values()) for value in result.values()) != count:
        raise AssertionError("quota arithmetic failed")
    return result


def is_ood_geometry(sx: int, sy: int, tx: int, ty: int, blocks: list[dict[str, int]],
                    boundary_margin: int, block_margin: int) -> bool:
    if min(sx, tx) <= boundary_margin or max(sx, tx) >= 119 - boundary_margin:
        return True
    if min(sy, ty) <= boundary_margin or max(sy, ty) >= 549 - boundary_margin:
        return True
    for block in blocks:
        near = any(
            abs(x - edge) <= block_margin
            for x in (sx, tx) for edge in (block["left"], block["right"])
        ) or any(
            abs(y - edge) <= block_margin
            for y in (sy, ty) for edge in (block["lower"], block["upper"])
        )
        horizontal_cross = (block["lower"] <= sy <= block["upper"] and
                            block["lower"] <= ty <= block["upper"] and
                            ((sx < block["left"] and tx > block["right"]) or
                             (tx < block["left"] and sx > block["right"])))
        vertical_cross = (block["left"] <= sx <= block["right"] and
                          block["left"] <= tx <= block["right"] and
                          ((sy < block["lower"] and ty > block["upper"]) or
                           (ty < block["lower"] and sy > block["upper"])))
        if near or horizontal_cross or vertical_cross:
            return True
    return False


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.template_rows <= 0 or args.boundary_margin < 0 or args.block_margin < 0:
        raise ValueError("count/template-rows must be positive and margins non-negative")
    answers = pd.read_csv(args.answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    if args.template_rows > len(answers):
        raise ValueError("--template-rows exceeds answer rows")
    source_info = {item: parse_endpoint(item) for item in answers["From"].drop_duplicates()}
    target_info = {item: parse_endpoint(item) for item in answers["To"].drop_duplicates()}
    sources_by_port: dict[str, list[str]] = defaultdict(list)
    targets_at_site: dict[tuple[int, int], set[str]] = defaultdict(set)
    for endpoint, (_, _, port) in source_info.items():
        sources_by_port[port].append(endpoint)
    for endpoint, (x, y, port) in target_info.items():
        targets_at_site[(x, y)].add(port)

    excluded = {endpoint_key(source, target) for source, target in zip(answers["From"], answers["To"])}
    exclusion_counts = {str(args.answers): len(excluded)}
    for path in candidate_files(args):
        frame = pd.read_csv(path, dtype={"From": "string", "To": "string"}, usecols=["From", "To"])
        before = len(excluded)
        excluded.update(endpoint_key(source, target) for source, target in zip(frame["From"], frame["To"]))
        exclusion_counts[str(path)] = len(excluded) - before

    template = answers.iloc[:args.template_rows]
    group_pool: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for source, target in zip(template["From"], template["To"]):
        sx, sy, source_port = source_info[source]
        tx, ty, target_port = target_info[target]
        dx, dy = tx - sx, ty - sy
        distance = abs(dx) + abs(dy)
        if distance <= 32:
            group_pool[layer_name(distance)].append((source_port, target_port, dx, dy))
    for name, _, _ in LAYERS:
        if not group_pool[name]:
            raise RuntimeError(f"no template group for layer {name}")

    with (args.arch / "SRB_Gap.json").open(encoding="utf-8") as stream:
        blocks = json.load(stream)["Gap"]["Block"]
    rng = np.random.default_rng(args.seed)
    selected_pairs: set[str] = set()
    selected_groups: dict[str, str] = {}
    records: list[dict[str, object]] = []
    requested = quotas(args.count)
    attempts: Counter[str] = Counter()

    for name, _, _ in LAYERS:
        pool = group_pool[name]
        for split in SPLITS:
            needed = requested[name][split]
            added = 0
            while added < needed:
                attempts[f"{name}:{split}"] += 1
                if attempts[f"{name}:{split}"] > max(100_000, needed * 1_000):
                    raise RuntimeError(f"could not fill {name}/{split}; inspect template coverage or exclusions")
                source_port, target_port, dx, dy = pool[int(rng.integers(len(pool)))]
                key = group_key(source_port, target_port, dx, dy)
                owner = selected_groups.get(key)
                if owner is None:
                    bucket = stable_bucket(args.seed, key)
                    intended = "dev" if bucket < 6 else "sealed_iid" if bucket < 8 else "sealed_ood"
                    if intended != split:
                        continue
                    selected_groups[key] = split
                elif owner != split:
                    continue
                source_pool = sources_by_port[source_port]
                source = source_pool[int(rng.integers(len(source_pool)))]
                sx, sy, _ = source_info[source]
                tx, ty = sx + dx, sy + dy
                if target_port not in targets_at_site.get((tx, ty), set()):
                    continue
                target = f"SRB_{tx}_{ty}/{target_port}"
                pair = endpoint_key(source, target)
                if pair in excluded or pair in selected_pairs:
                    continue
                stress = is_ood_geometry(sx, sy, tx, ty, blocks, args.boundary_margin, args.block_margin)
                if split == "sealed_ood" and not stress:
                    continue
                selected_pairs.add(pair)
                records.append({"From": source, "To": target, "split": split, "stratum": name,
                                "group_key": key, "source_port": source_port, "target_port": target_port,
                                "dx": dx, "dy": dy, "ood_stress": int(stress)})
                added += 1

    generated = pd.DataFrame(records).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        portion = generated.loc[generated["split"] == split].copy()
        portion[["From", "To"]].to_csv(args.output_dir / f"{split}.csv", index=False)
        portion.to_csv(args.output_dir / f"{split}.metadata.csv", index=False)

    # This is public-answer validation material only. It contains no candidates.
    public_rows = []
    infos_from = answers["From"].map(source_info)
    infos_to = answers["To"].map(target_info)
    distances = np.fromiter((abs(target[0] - source[0]) + abs(target[1] - source[1])
                             for source, target in zip(infos_from, infos_to)), dtype=np.int16)
    for number, (name, lower, upper) in enumerate(LAYERS):
        indices = np.flatnonzero((distances >= lower) & (distances <= upper))
        # The public corpus has only 22 Manhattan-0 rows.  Preserve the six
        # 100-row oracle checks by sampling that stratum with replacement and
        # disclose the reduced unique coverage in the manifest.
        chosen = np.random.default_rng(args.seed + number).choice(
            indices, size=args.public_check_per_layer, replace=len(indices) < args.public_check_per_layer
        )
        sample = answers.iloc[np.sort(chosen)][["From", "To", "delay"]].copy()
        sample.insert(2, "stratum", name)
        public_rows.append(sample)
    public_check = pd.concat(public_rows, ignore_index=True)
    public_check[["From", "To"]].to_csv(args.output_dir / "public_exact_check.csv", index=False)
    public_check.to_csv(args.output_dir / "public_exact_check.expected.csv", index=False)

    report = {
        "schema_version": 1, "count": int(len(generated)), "seed": args.seed,
        "template_rows": args.template_rows, "splits": generated["split"].value_counts().sort_index().to_dict(),
        "strata": generated.groupby(["split", "stratum"]).size().unstack(fill_value=0).to_dict(),
        "unique_pairs": int(len(selected_pairs)), "unique_group_keys": int(generated["group_key"].nunique()),
        "excluded_pairs": int(len(excluded)), "exclusion_new_pairs": exclusion_counts,
        "sealed_policy": "Label sealed_iid and sealed_ood only after configuration freeze; do not open labels or metrics during tuning.",
        "ood_definition": "endpoint is near a chip boundary or Block boundary, or crosses a Block band",
        "public_exact_check_rows": int(len(public_check)),
        "public_exact_check_unique_pairs": {
            name: int(public_check.loc[public_check["stratum"] == name, ["From", "To"]].drop_duplicates().shape[0])
            for name, _, _ in LAYERS
        },
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
