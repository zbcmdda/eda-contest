"""Python reference for the bounded offline C++ short-route descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from train_model import ArchitectureFeatures, endpoint_columns


TWO_HOP_NAMES = [
    "a1_legal", "a1_blocked", "a1_duplicates", "a1_best", "a1_second", "a1_third",
    "a1_gap", "a1_near_best", "a1_directions", "a1_best_direction", "a1_toward_gap",
    "a1_best_reverse", "a1_best_overshoot", "a1_best_turn", "a1_topology_penalty",
    "a1_complete_cost", "a1_exact", "b2_legal", "b2_best", "b2_second", "b2_gap",
    "b2_complete_cost", "b2_exact", "b2_states",
]
THREE_HOP_NAMES = [
    "b3_legal", "b3_best", "b3_second", "b3_gap", "b3_complete_cost", "b3_exact",
    "b3_states",
]
NAMES = TWO_HOP_NAMES + THREE_HOP_NAMES
MISSING = -1


class ShortFeatureReference:
    def __init__(self, architecture: ArchitectureFeatures):
        self.a = architecture
        self.final = {name: (0 if value == 65_535 else int(value)) for name, value in architecture.final_delay.items()}
        self.net = architecture.net_by_from

    def _resolve(self, x: int, y: int, net: dict) -> tuple[int, int, int] | None:
        xs = np.asarray([x], dtype=np.int16)
        ys = np.asarray([y], dtype=np.int16)
        nx, ny, delay, valid = self.a.resolve_net_arrays(xs, ys, net)
        return (int(nx[0]), int(ny[0]), int(delay[0])) if valid[0] else None

    def _heuristic(self, state: tuple[int, int, str, int, int], target: tuple[int, int, str]) -> int:
        x, y, _, _, _ = state
        tx, ty, target_port = target
        dx, dy = tx - x, ty - y
        return int(self.a.relaxed_x[dx + self.a.width - 1] + self.a.relaxed_y[dy + self.a.height - 1]
                   + abs(int(self.a.vertical_prefix[tx]) - int(self.a.vertical_prefix[x]))
                   + abs(int(self.a.horizontal_prefix[ty]) - int(self.a.horizontal_prefix[y]))
                   + self.final[target_port])

    def _complete(self, state: tuple[int, int, str, int, int], target: tuple[int, int, str]) -> int:
        x, y, port, g, _ = state
        tx, ty, target_port = target
        if x != tx or y != ty:
            return MISSING
        arc = next((arc["delay"] for arc in self.a.arcs if arc["from"] == port and arc["to"] == target_port), None)
        return MISSING if arc is None else int(g + arc)

    def _expand(self, state: tuple[int, int, str, int, int], target: tuple[int, int, str],
                output: list[tuple[int, int, str, int, int]], counts: list[int]) -> None:
        x, y, port, g, _ = state
        for arc, net in self.a.routing_arcs_by_from.get(port, []):
            resolved = self._resolve(x, y, net)
            if resolved is None:
                counts[1] += 1
                continue
            counts[0] += 1
            nx, ny, delay = resolved
            direction = 0 if net["delta x"] > 0 else 1 if net["delta x"] < 0 else 2 if net["delta y"] > 0 else 3
            next_state = (nx, ny, net["to"], int(g + arc["delay"] + delay), direction)
            same = next((index for index, current in enumerate(output)
                         if current[:3] == next_state[:3]), None)
            if same is not None:
                counts[2] += 1
                if (next_state[3], next_state[4]) < (output[same][3], output[same][4]):
                    output[same] = next_state
            else:
                output.append(next_state)

    def row(self, source: tuple[int, int, str], target: tuple[int, int, str]) -> list[int]:
        sx, sy, source_port = source
        start = (sx, sy, source_port, 0, -1)
        source_net = self.net.get(source_port)
        if source_net is not None and self.a.port_direction[source_port] == "Output":
            resolved = self._resolve(sx, sy, source_net)
            if resolved is not None:
                nx, ny, delay = resolved
                start = (nx, ny, source_net["to"], delay, -1)
        first: list[tuple[int, int, str, int, int]] = []
        counts = [0, 0, 0]
        self._expand(start, target, first, counts)
        first.sort(key=lambda state: (
            state[3] + self._heuristic(state, target), state[3], state[0], state[1],
            self.a.port_id[state[2]], state[4]))
        values = [MISSING] * len(NAMES)
        values[:3] = counts
        if not first:
            return values
        estimates = [state[3] + self._heuristic(state, target) for state in first]
        values[3] = estimates[0]
        for output, input_index in ((4, 1), (5, 2)):
            if len(first) > input_index: values[output] = estimates[input_index]
        values[6] = estimates[1] - estimates[0] if len(first) > 1 else MISSING
        values[7] = sum(value <= estimates[0] + 50 for value in estimates)
        values[8] = len({state[4] for state in first})
        values[9] = first[0][4]
        dx, dy = target[0] - start[0], target[1] - start[1]
        toward = 0 if dx > 0 else 1 if dx < 0 else 2 if dy > 0 else 3 if dy < 0 else -1
        toward_values = [estimate for state, estimate in zip(first, estimates) if state[4] == toward]
        values[10] = min(toward_values) - estimates[0] if toward_values else MISSING
        values[11] = int(toward >= 0 and first[0][4] != toward)
        values[12] = int(abs(target[0] - first[0][0]) + abs(target[1] - first[0][1]) > abs(dx) + abs(dy))
        values[13] = values[11]
        values[14] = estimates[0] - (start[3] + self._heuristic(start, target))
        complete = self._complete(start, target)
        for state in first:
            candidate = self._complete(state, target)
            if candidate >= 0 and (complete < 0 or candidate < complete): complete = candidate
        values[15], values[16] = complete, int(complete >= 0)
        second: list[tuple[int, int, str, int, int]] = []
        second_counts = [0, 0, 0]
        for state in first[:4]: self._expand(state, target, second, second_counts)
        second.sort(key=lambda state: (
            state[3] + self._heuristic(state, target), state[3], state[0], state[1],
            self.a.port_id[state[2]], state[4]))
        values[17], values[23] = second_counts[0], len(second)
        if second:
            estimates2 = [state[3] + self._heuristic(state, target) for state in second]
            values[18] = estimates2[0]
            if len(second) > 1:
                values[19], values[20] = estimates2[1], estimates2[1] - estimates2[0]
            for state in second:
                candidate = self._complete(state, target)
                if candidate >= 0 and (complete < 0 or candidate < complete): complete = candidate
        values[21], values[22] = complete, int(complete >= 0)
        third: list[tuple[int, int, str, int, int]] = []
        third_counts = [0, 0, 0]
        for state in second[:2]: self._expand(state, target, third, third_counts)
        third.sort(key=lambda state: (
            state[3] + self._heuristic(state, target), state[3], state[0], state[1],
            self.a.port_id[state[2]], state[4]))
        values[24], values[30] = third_counts[0], len(third)
        if third:
            estimates3 = [state[3] + self._heuristic(state, target) for state in third]
            values[25] = estimates3[0]
            if len(third) > 1: values[26], values[27] = estimates3[1], estimates3[1] - estimates3[0]
            for state in third:
                candidate = self._complete(state, target)
                if candidate >= 0 and (complete < 0 or candidate < complete): complete = candidate
        values[28], values[29] = complete, int(complete >= 0)
        return values

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        source, target = endpoint_columns(frame["From"], "source"), endpoint_columns(frame["To"], "target")
        rows = [self.row((int(sx), int(sy), sp), (int(tx), int(ty), tp))
                for sx, sy, sp, tx, ty, tp in zip(source.source_x, source.source_y, source.source_name,
                                                   target.target_x, target.target_y, target.target_name)]
        return pd.DataFrame(rows, columns=NAMES, index=frame.index, dtype=np.int32)
