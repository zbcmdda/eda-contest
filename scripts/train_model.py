#!/usr/bin/env python3
"""Train and evaluate a compact LightGBM delay estimator."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--answers", type=Path, default=Path("data/delay_estimate_ans.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/delay_model.txt"))
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--leaves", type=int, default=127)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--validation-rows", type=int, default=100_000)
    parser.add_argument("--training-rows", type=int)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--filter-max-relaxed", type=int)
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
    parser.add_argument(
        "--objective",
        choices=("log_l2", "log_l1", "l1", "mape", "residual_log_l2"),
        default="log_l2",
    )
    return parser.parse_args()


def load_json(path: Path, key: str):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)[key]


def endpoint_columns(series: pd.Series, prefix: str) -> pd.DataFrame:
    parts = series.str.extract(r"^SRB_(\d+)_(\d+)/(.*)$")
    if parts.isna().any().any():
        raise ValueError(f"malformed {prefix} endpoint")
    return pd.DataFrame(
        {
            f"{prefix}_x": parts[0].astype(np.int16),
            f"{prefix}_y": parts[1].astype(np.int16),
            f"{prefix}_name": parts[2],
        }
    )


def relaxed_axis_distances(maximum_delta: int, moves: list[tuple[int, int]]) -> np.ndarray:
    maximum_move = max(abs(delta) for delta, _ in moves)
    extent = maximum_delta + maximum_move
    infinity = np.iinfo(np.int32).max
    distances = np.full(2 * extent + 1, infinity, dtype=np.int32)
    distances[extent] = 0
    queue: list[tuple[int, int]] = [(0, 0)]
    while queue:
        distance, coordinate = heapq.heappop(queue)
        if distance != distances[coordinate + extent]:
            continue
        for delta, cost in moves:
            target = coordinate + delta
            if target < -extent or target > extent:
                continue
            candidate = distance + cost
            if candidate < distances[target + extent]:
                distances[target + extent] = candidate
                heapq.heappush(queue, (candidate, target))
    return distances[maximum_move : maximum_move + 2 * maximum_delta + 1]


class ArchitectureFeatures:
    def __init__(self, arch: Path):
        with (arch / "SRB_Inst.json").open(encoding="utf-8") as stream:
            instances = json.load(stream)["Inst"]
        self.width = max(instance["x"] for instance in instances) + 1
        self.height = max(instance["y"] for instance in instances) + 1
        self.ports = load_json(arch / "SRB_Port.json", "Port")
        self.arcs = load_json(arch / "SRB_Arc.json", "Arcs")
        self.nets = load_json(arch / "SRB_Net.json", "Nets")
        with (arch / "SRB_Gap.json").open(encoding="utf-8") as stream:
            gaps = json.load(stream)["Gap"]
        self.lines = gaps["Line"]
        self.blocks = gaps["Block"]

        active = np.zeros((self.height, self.width), dtype=bool)
        for instance in instances:
            active[instance["y"], instance["x"]] = True
        self.active = active
        self.clearance = self._build_clearance(active)

        self.port_id = {port["Name"]: index for index, port in enumerate(self.ports)}
        self.port_direction = {port["Name"]: port["Direction"] for port in self.ports}
        self.net_by_from = {net["from"]: net for net in self.nets}
        self.routing_arcs_by_from: dict[str, list[tuple[dict, dict]]] = {}
        for arc in self.arcs:
            net = self.net_by_from.get(arc["to"])
            if net is not None:
                self.routing_arcs_by_from.setdefault(arc["from"], []).append((arc, net))

        self.final_delay = {name: 65_535 for name in self.port_id}
        self.first_routing_delay = {name: 65_535 for name in self.port_id}
        minimum_move_cost: dict[tuple[int, int], int] = {}
        for arc in self.arcs:
            self.final_delay[arc["to"]] = min(self.final_delay[arc["to"]], arc["delay"])
            net = self.net_by_from.get(arc["to"])
            if net is not None:
                self.first_routing_delay[arc["from"]] = min(
                    self.first_routing_delay[arc["from"]], arc["delay"]
                )
                movement = (net["delta x"], net["delta y"])
                minimum_move_cost[movement] = min(minimum_move_cost.get(movement, 65_535), arc["delay"])

        horizontal_moves: list[tuple[int, int]] = []
        vertical_moves: list[tuple[int, int]] = []
        for (dx, dy), cost in minimum_move_cost.items():
            moves = horizontal_moves if dx else vertical_moves
            delta = dx or dy
            moves.append((delta, cost))
            for block in self.blocks:
                horizontal = bool(dx)
                if not block["horizontal crossable" if horizontal else "vertical crossable"]:
                    continue
                size = (
                    block["right"] - block["left"] + 1
                    if horizontal
                    else block["upper"] - block["lower"] + 1
                )
                delay = block["horizontal cross delay" if horizontal else "vertical cross delay"]
                moves.append((delta + (size if delta > 0 else -size), cost + delay))
        self.relaxed_x = relaxed_axis_distances(119, horizontal_moves)
        self.relaxed_y = relaxed_axis_distances(549, vertical_moves)

        self.vertical_prefix = np.zeros(120, dtype=np.int16)
        self.horizontal_prefix = np.zeros(550, dtype=np.int16)
        for coordinate in range(120):
            self.vertical_prefix[coordinate] = sum(
                line["delay"]
                for line in self.lines
                if line["direction"] == "vertical" and line["site"] < coordinate
            )
        for coordinate in range(550):
            self.horizontal_prefix[coordinate] = sum(
                line["delay"]
                for line in self.lines
                if line["direction"] == "horizontal" and line["site"] < coordinate
            )

    @staticmethod
    def _build_clearance(active: np.ndarray) -> dict[str, np.ndarray]:
        """Count consecutive active SRBs before a Block or array boundary."""
        height, width = active.shape
        result = {
            "east": np.zeros_like(active, dtype=np.int16),
            "west": np.zeros_like(active, dtype=np.int16),
            "north": np.zeros_like(active, dtype=np.int16),
            "south": np.zeros_like(active, dtype=np.int16),
        }
        for x in range(width - 2, -1, -1):
            result["east"][:, x] = np.where(
                active[:, x + 1], result["east"][:, x + 1] + 1, 0
            )
        for x in range(1, width):
            result["west"][:, x] = np.where(
                active[:, x - 1], result["west"][:, x - 1] + 1, 0
            )
        for y in range(height - 2, -1, -1):
            result["north"][y, :] = np.where(
                active[y + 1, :], result["north"][y + 1, :] + 1, 0
            )
        for y in range(1, height):
            result["south"][y, :] = np.where(
                active[y - 1, :], result["south"][y - 1, :] + 1, 0
            )
        return result

    def local_topology_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        prefix: str,
    ) -> pd.DataFrame:
        columns = {}
        for direction in ("east", "west", "north", "south"):
            columns[f"{prefix}_{direction}_clearance"] = self.clearance[direction][y, x]
        return pd.DataFrame(columns)

    def resolve_net_arrays(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        net: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized counterpart of Architecture::resolve_net for one Net template."""
        dx = int(net["delta x"])
        dy = int(net["delta y"])
        target_x = source_x.astype(np.int16, copy=True) + dx
        target_y = source_y.astype(np.int16, copy=True) + dy
        delay = np.zeros(len(source_x), dtype=np.int32)

        for block in self.blocks:
            horizontal = dx != 0
            if not block["horizontal crossable" if horizontal else "vertical crossable"]:
                continue
            if horizontal:
                in_band = (source_y >= block["lower"]) & (source_y <= block["upper"])
                size = block["right"] - block["left"] + 1
                if dx > 0:
                    crosses = in_band & (source_x < block["left"]) & (target_x >= block["left"])
                    target_x[crosses] += size
                else:
                    crosses = in_band & (source_x > block["right"]) & (target_x <= block["right"])
                    target_x[crosses] -= size
                delay[crosses] += block["horizontal cross delay"]
            else:
                in_band = (source_x >= block["left"]) & (source_x <= block["right"])
                size = block["upper"] - block["lower"] + 1
                if dy > 0:
                    crosses = in_band & (source_y < block["lower"]) & (target_y >= block["lower"])
                    target_y[crosses] += size
                else:
                    crosses = in_band & (source_y > block["upper"]) & (target_y <= block["upper"])
                    target_y[crosses] -= size
                delay[crosses] += block["vertical cross delay"]

        valid = (
            (target_x >= 0)
            & (target_x < self.width)
            & (target_y >= 0)
            & (target_y < self.height)
        )
        safe_x = np.clip(target_x, 0, self.width - 1)
        safe_y = np.clip(target_y, 0, self.height - 1)
        valid &= self.active[safe_y, safe_x]
        delay += np.abs(self.vertical_prefix[safe_x] - self.vertical_prefix[source_x])
        delay += np.abs(self.horizontal_prefix[safe_y] - self.horizontal_prefix[source_y])
        return target_x, target_y, delay, valid

    def first_step_lookahead_features(
        self,
        source_net: pd.DataFrame,
        effective_source_name: pd.Series,
        target: pd.DataFrame,
    ) -> pd.DataFrame:
        row_count = len(target)
        missing = np.int32(1_000_000_000)
        directional = np.full((4, row_count), missing, dtype=np.int32)
        source_x_all = source_net["after_source_x"].to_numpy(dtype=np.int16)
        source_y_all = source_net["after_source_y"].to_numpy(dtype=np.int16)
        source_delay_all = source_net["source_net_delay"].to_numpy(dtype=np.int32)
        target_x_all = target["target_x"].to_numpy(dtype=np.int16)
        target_y_all = target["target_y"].to_numpy(dtype=np.int16)
        final_delay_all = target["target_name"].map(self.final_delay).to_numpy(dtype=np.int32)
        final_delay_all = np.where(final_delay_all == 65_535, 0, final_delay_all)
        effective = effective_source_name.to_numpy(dtype=object)

        for source_name, routing_arcs in self.routing_arcs_by_from.items():
            rows = np.flatnonzero(effective == source_name)
            if len(rows) == 0:
                continue
            source_x = source_x_all[rows]
            source_y = source_y_all[rows]
            target_x = target_x_all[rows]
            target_y = target_y_all[rows]
            for arc, net in routing_arcs:
                next_x, next_y, net_delay, valid = self.resolve_net_arrays(source_x, source_y, net)
                if not valid.any():
                    continue
                local = np.flatnonzero(valid)
                global_rows = rows[local]
                residual_x = target_x[local] - next_x[local]
                residual_y = target_y[local] - next_y[local]
                candidate = (
                    source_delay_all[global_rows]
                    + int(arc["delay"])
                    + net_delay[local]
                    + self.relaxed_x[residual_x + self.width - 1]
                    + self.relaxed_y[residual_y + self.height - 1]
                    + np.abs(
                        self.vertical_prefix[target_x[local]]
                        - self.vertical_prefix[next_x[local]]
                    )
                    + np.abs(
                        self.horizontal_prefix[target_y[local]]
                        - self.horizontal_prefix[next_y[local]]
                    )
                    + final_delay_all[global_rows]
                ).astype(np.int32)
                dx = int(net["delta x"])
                dy = int(net["delta y"])
                direction = 0 if dx > 0 else 1 if dx < 0 else 2 if dy > 0 else 3
                directional[direction, global_rows] = np.minimum(
                    directional[direction, global_rows], candidate
                )

        all_min = directional.min(axis=0)
        horizontal_min = directional[:2].min(axis=0)
        vertical_min = directional[2:].min(axis=0)
        dx = target_x_all - source_x_all
        dy = target_y_all - source_y_all
        toward_x = np.where(dx > 0, directional[0], np.where(dx < 0, directional[1], horizontal_min))
        toward_y = np.where(dy > 0, directional[2], np.where(dy < 0, directional[3], vertical_min))
        toward = np.minimum(toward_x, toward_y)

        def expose(values: np.ndarray) -> np.ndarray:
            return np.where(values == missing, -1, values)

        return pd.DataFrame(
            {
                "lookahead_east": expose(directional[0]),
                "lookahead_west": expose(directional[1]),
                "lookahead_north": expose(directional[2]),
                "lookahead_south": expose(directional[3]),
                "lookahead_min": expose(all_min),
                "lookahead_toward": expose(toward),
            }
        )

    def source_net_features(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        source_x = source["source_x"].to_numpy(dtype=np.int16)
        source_y = source["source_y"].to_numpy(dtype=np.int16)
        net_x = source_x.copy()
        net_y = source_y.copy()
        block_delay = np.zeros(len(source), dtype=np.int16)
        effective_name = source["source_name"].copy()

        for name, net in self.net_by_from.items():
            selected = (source["source_name"] == name).fillna(False).to_numpy(dtype=bool)
            if not selected.any():
                continue
            effective_name.loc[selected] = net["to"]
            dx = net["delta x"]
            dy = net["delta y"]
            net_x[selected] += dx
            net_y[selected] += dy
            for block in self.blocks:
                horizontal = dx != 0
                if not block["horizontal crossable" if horizontal else "vertical crossable"]:
                    continue
                if horizontal:
                    in_band = (source_y >= block["lower"]) & (source_y <= block["upper"])
                    size = block["right"] - block["left"] + 1
                    if dx > 0:
                        crosses = selected & in_band & (source_x < block["left"]) & (net_x >= block["left"])
                        net_x[crosses] += size
                    else:
                        crosses = selected & in_band & (source_x > block["right"]) & (net_x <= block["right"])
                        net_x[crosses] -= size
                    block_delay[crosses] += block["horizontal cross delay"]
                else:
                    in_band = (source_x >= block["left"]) & (source_x <= block["right"])
                    size = block["upper"] - block["lower"] + 1
                    if dy > 0:
                        crosses = selected & in_band & (source_y < block["lower"]) & (net_y >= block["lower"])
                        net_y[crosses] += size
                    else:
                        crosses = selected & in_band & (source_y > block["upper"]) & (net_y <= block["upper"])
                        net_y[crosses] -= size
                    block_delay[crosses] += block["vertical cross delay"]

        line_delay = np.abs(self.vertical_prefix[net_x] - self.vertical_prefix[source_x])
        line_delay += np.abs(self.horizontal_prefix[net_y] - self.horizontal_prefix[source_y])
        after_dx = target["target_x"].to_numpy(dtype=np.int16) - net_x
        after_dy = target["target_y"].to_numpy(dtype=np.int16) - net_y
        result = pd.DataFrame(
            {
                "after_source_x": net_x,
                "after_source_y": net_y,
                "after_source_dx": after_dx,
                "after_source_dy": after_dy,
                "after_source_absolute_x": np.abs(after_dx),
                "after_source_absolute_y": np.abs(after_dy),
                "source_net_delay": block_delay + line_delay,
                "effective_source_port": effective_name.map(self.port_id).astype(np.int16).astype("category"),
            },
            index=source.index,
        )
        return result, effective_name

    @staticmethod
    def port_semantics(names: pd.Series, prefix: str) -> pd.DataFrame:
        first = names.str[0]
        inter = first.isin(["I", "Z"])
        span_text = names.str[1].where(inter, "X")
        span = span_text.map({"S": 1, "D": 2, "Q": 4, "L": 10, "X": 0}).fillna(0).astype(np.int8)
        direction = names.str.extract(r"^[IZ][SDQL]([NSEW])", expand=False).fillna("X")
        bit = pd.to_numeric(names.str.extract(r"\[(\d+)\]", expand=False), errors="coerce").fillna(-1)
        return pd.DataFrame(
            {
                f"{prefix}_kind": first.astype("category"),
                f"{prefix}_span_kind": span_text.astype("category"),
                f"{prefix}_span": span,
                f"{prefix}_direction": direction.astype("category"),
                f"{prefix}_bit": bit.astype(np.int8),
                f"{prefix}_inter": inter.astype(np.int8),
            }
        )

    def build(
        self,
        raw: pd.DataFrame,
        include_topology: bool = False,
        include_lookahead: bool = False,
    ) -> pd.DataFrame:
        source = endpoint_columns(raw["From"], "source")
        target = endpoint_columns(raw["To"], "target")
        features = pd.DataFrame(index=raw.index)
        for column in ("source_x", "source_y"):
            features[column] = source[column]
        for column in ("target_x", "target_y"):
            features[column] = target[column]

        dx = target["target_x"].astype(np.int16) - source["source_x"].astype(np.int16)
        dy = target["target_y"].astype(np.int16) - source["source_y"].astype(np.int16)
        absolute_x = dx.abs().astype(np.int16)
        absolute_y = dy.abs().astype(np.int16)
        features["dx"] = dx
        features["dy"] = dy
        features["absolute_x"] = absolute_x
        features["absolute_y"] = absolute_y
        features["manhattan"] = absolute_x + absolute_y
        features["x_long_count"] = absolute_x // 10
        features["x_long_remainder"] = (absolute_x % 10).astype("category")
        features["y_long_count"] = absolute_y // 12
        features["y_long_remainder"] = (absolute_y % 12).astype("category")
        features["x_sign"] = np.sign(dx).astype(np.int8).astype("category")
        features["y_sign"] = np.sign(dy).astype(np.int8).astype("category")

        source_ids = source["source_name"].map(self.port_id)
        target_ids = target["target_name"].map(self.port_id)
        if source_ids.isna().any() or target_ids.isna().any():
            raise ValueError("answer CSV references an unknown port")
        features["source_port"] = source_ids.astype(np.int16).astype("category")
        features["target_port"] = target_ids.astype(np.int16).astype("category")
        features = pd.concat(
            [
                features,
                self.port_semantics(source["source_name"], "source"),
                self.port_semantics(target["target_name"], "target"),
            ],
            axis=1,
        )

        source_net, effective_source_name = self.source_net_features(source, target)
        features = pd.concat([features, source_net], axis=1)
        if include_topology:
            source_x = source["source_x"].to_numpy(dtype=np.int16)
            source_y = source["source_y"].to_numpy(dtype=np.int16)
            target_x = target["target_x"].to_numpy(dtype=np.int16)
            target_y = target["target_y"].to_numpy(dtype=np.int16)
            after_x = source_net["after_source_x"].to_numpy(dtype=np.int16)
            after_y = source_net["after_source_y"].to_numpy(dtype=np.int16)
            topology = pd.concat(
                [
                    self.local_topology_features(source_x, source_y, "source"),
                    self.local_topology_features(target_x, target_y, "target"),
                    self.local_topology_features(after_x, after_y, "after_source"),
                ],
                axis=1,
            )
            features = pd.concat([features, topology], axis=1)
        if include_lookahead:
            features = pd.concat(
                [
                    features,
                    self.first_step_lookahead_features(
                        source_net, effective_source_name, target
                    ),
                ],
                axis=1,
            )
        endpoint_pair = (
            effective_source_name.map(self.port_id).astype(np.int32) * len(self.ports)
            + target_ids.astype(np.int32)
        )
        features["endpoint_pair"] = endpoint_pair.astype("category")

        line_x = np.abs(
            self.vertical_prefix[target["target_x"].to_numpy()]
            - self.vertical_prefix[source["source_x"].to_numpy()]
        )
        line_y = np.abs(
            self.horizontal_prefix[target["target_y"].to_numpy()]
            - self.horizontal_prefix[source["source_y"].to_numpy()]
        )
        features["line_x_delay"] = line_x
        features["line_y_delay"] = line_y
        relaxed = (
            self.relaxed_x[dx.to_numpy() + 119]
            + self.relaxed_y[dy.to_numpy() + 549]
            + line_x
            + line_y
        )
        final = target["target_name"].map(self.final_delay).to_numpy()
        final = np.where(final == 65_535, 0, final)
        first = source["source_name"].map(self.first_routing_delay).to_numpy()
        first = np.where(first == 65_535, 0, first)
        features["relaxed_delay"] = relaxed
        features["target_final_min"] = final
        features["source_first_min"] = first
        after_dx = source_net["after_source_dx"].to_numpy()
        after_dy = source_net["after_source_dy"].to_numpy()
        after_x = source_net["after_source_x"].to_numpy()
        after_y = source_net["after_source_y"].to_numpy()
        after_line_x = np.abs(self.vertical_prefix[target["target_x"].to_numpy()] - self.vertical_prefix[after_x])
        after_line_y = np.abs(self.horizontal_prefix[target["target_y"].to_numpy()] - self.horizontal_prefix[after_y])
        features["relaxed_after_source"] = (
            self.relaxed_x[after_dx + 119]
            + self.relaxed_y[after_dy + 549]
            + after_line_x
            + after_line_y
            + source_net["source_net_delay"].to_numpy()
            + final
        )

        for block in self.blocks:
            block_id = block["id"]
            source_band = source["source_y"].between(block["lower"], block["upper"])
            target_band = target["target_y"].between(block["lower"], block["upper"])
            left_to_right = (source["source_x"] < block["left"]) & (target["target_x"] > block["right"])
            right_to_left = (source["source_x"] > block["right"]) & (target["target_x"] < block["left"])
            features[f"block_{block_id}_same_band_cross"] = (
                source_band & target_band & (left_to_right | right_to_left)
            ).astype(np.int8)
        return features


def official_metrics(golden: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    positive = golden > 0
    relative = np.abs(predicted[positive] - golden[positive]) / golden[positive]
    return {
        "official_score": float(np.mean(1.0 - np.tanh(4.0 * relative)) * 100.0),
        "mean_relative_error": float(np.mean(relative)),
        "median_relative_error": float(np.median(relative)),
        "within_5_percent": float(np.mean(relative <= 0.05) * 100.0),
        "within_10_percent": float(np.mean(relative <= 0.10) * 100.0),
        "within_20_percent": float(np.mean(relative <= 0.20) * 100.0),
    }


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.answers, dtype={"From": "string", "To": "string", "delay": np.int32})
    if args.validation_rows <= 0 or args.validation_rows >= len(raw):
        raise ValueError("validation row count must be between zero and the dataset size")
    architecture = ArchitectureFeatures(args.arch)
    features = architecture.build(
        raw,
        include_topology=args.topology_features,
        include_lookahead=args.lookahead_features,
    )
    categorical = [column for column in features if isinstance(features[column].dtype, pd.CategoricalDtype)]

    split = len(raw) - args.validation_rows
    train_x = features.iloc[:split]
    valid_x = features.iloc[split:]
    train_y_raw = raw["delay"].to_numpy(dtype=np.float64)[:split]
    valid_y = raw["delay"].to_numpy(dtype=np.float64)[split:]
    if args.training_rows is not None:
        if args.training_rows <= 0 or args.training_rows > len(train_x):
            raise ValueError("training row count must be positive and no larger than the train split")
        train_x = train_x.iloc[: args.training_rows]
        train_y_raw = train_y_raw[: args.training_rows]
    if args.filter_max_relaxed is not None:
        train_selected = train_x["relaxed_after_source"].to_numpy() <= args.filter_max_relaxed
        valid_selected = valid_x["relaxed_after_source"].to_numpy() <= args.filter_max_relaxed
        train_x = train_x.loc[train_selected]
        valid_x = valid_x.loc[valid_selected]
        train_y_raw = train_y_raw[train_selected]
        valid_y = valid_y[valid_selected]
    if args.objective == "residual_log_l2":
        train_baseline = train_x["relaxed_after_source"].to_numpy(dtype=np.float64)
        train_y = np.log((train_y_raw + 1.0) / (train_baseline + 1.0))
    else:
        train_y = np.log1p(train_y_raw) if args.objective.startswith("log_") else train_y_raw
    objective = {
        "log_l2": "regression",
        "log_l1": "regression_l1",
        "l1": "regression_l1",
        "mape": "mape",
        "residual_log_l2": "regression",
    }[args.objective]

    train_set = lgb.Dataset(train_x, label=train_y, categorical_feature=categorical, free_raw_data=False)
    parameters = {
        "objective": objective,
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
    model = lgb.train(parameters, train_set, num_boost_round=args.trees)
    predicted = model.predict(valid_x)
    if args.objective == "residual_log_l2":
        baseline = valid_x["relaxed_after_source"].to_numpy(dtype=np.float64)
        predicted = (baseline + 1.0) * np.exp(predicted) - 1.0
    elif args.objective.startswith("log_"):
        predicted = np.expm1(predicted)
    predicted = np.maximum(0.0, predicted)
    predicted[valid_y == 0] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    metadata = {
        "feature_names": list(features.columns),
        "categorical_features": categorical,
        "objective": args.objective,
        "trees": args.trees,
        "leaves": args.leaves,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "training_rows": len(train_x),
        "validation_rows": len(valid_x),
        "filter_max_relaxed": args.filter_max_relaxed,
        "metrics": official_metrics(valid_y, predicted),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
