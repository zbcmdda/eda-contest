#include "short_features.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace delay {
namespace {

constexpr std::int32_t kMissing = -1;
constexpr std::int32_t kInfinity = std::numeric_limits<std::int32_t>::max() / 4;

std::vector<std::uint32_t> relaxed_distances(int maximum_delta,
                                             const std::vector<std::pair<int, int>>& moves) {
    const int maximum_move = [&moves] {
        int value = 1;
        for (const auto& move : moves) value = std::max(value, std::abs(move.first));
        return value;
    }();
    const int extent = maximum_delta + maximum_move;
    std::vector<std::uint32_t> distance(static_cast<std::size_t>(2 * extent + 1),
                                        std::numeric_limits<std::uint32_t>::max());
    distance[static_cast<std::size_t>(extent)] = 0;
    // Relaxation is intentionally simple here: the architecture has only a
    // handful of signed moves, and construction is offline/startup-only.
    bool changed = true;
    while (changed) {
        changed = false;
        for (int position = -extent; position <= extent; ++position) {
            const auto base = distance[static_cast<std::size_t>(position + extent)];
            if (base == std::numeric_limits<std::uint32_t>::max()) continue;
            for (const auto& move : moves) {
                const int next = position + move.first;
                if (next < -extent || next > extent) continue;
                const auto candidate = base + static_cast<std::uint32_t>(move.second);
                auto& known = distance[static_cast<std::size_t>(next + extent)];
                if (candidate < known) { known = candidate; changed = true; }
            }
        }
    }
    std::vector<std::uint32_t> result(static_cast<std::size_t>(2 * maximum_delta + 1));
    for (int delta = -maximum_delta; delta <= maximum_delta; ++delta) {
        result[static_cast<std::size_t>(delta + maximum_delta)] =
            distance[static_cast<std::size_t>(delta + extent)];
    }
    return result;
}

}  // namespace

ShortFeatures::ShortFeatures(const Architecture& architecture) : architecture_(architecture) {
    port_count_ = architecture.ports().size();
    routing_arcs_.resize(port_count_);
    arc_delays_.assign(port_count_ * port_count_, std::numeric_limits<std::uint16_t>::max());
    std::vector<std::int8_t> net_directions(port_count_, -1);
    std::vector<std::int16_t> net_indices(port_count_, -1);
    for (std::size_t index = 0; index < architecture.nets().size(); ++index) {
        const auto& net = architecture.nets()[index];
        net_directions[net.from] = static_cast<std::int8_t>(
            net.dx > 0 ? 0 : net.dx < 0 ? 1 : net.dy > 0 ? 2 : 3);
        net_indices[net.from] = static_cast<std::int16_t>(index);
    }
    const auto build_band_classes = [&architecture](bool horizontal) {
        const int extent = horizontal ? architecture.height() : architecture.width();
        std::array<std::int16_t, 128> mask_to_class{};
        mask_to_class.fill(-1);
        std::vector<std::uint8_t> classes(static_cast<std::size_t>(extent));
        std::vector<std::int16_t> representatives;
        for (int coordinate = 0; coordinate < extent; ++coordinate) {
            std::uint8_t mask = 0;
            for (std::size_t index = 0; index < architecture.block_gaps().size(); ++index) {
                const auto& block = architecture.block_gaps()[index];
                const bool inside = horizontal
                                        ? coordinate >= block.lower && coordinate <= block.upper
                                        : coordinate >= block.left && coordinate <= block.right;
                if (inside) mask |= static_cast<std::uint8_t>(1u << index);
            }
            auto& class_id = mask_to_class[mask];
            if (class_id < 0) {
                class_id = static_cast<std::int16_t>(representatives.size());
                representatives.push_back(static_cast<std::int16_t>(coordinate));
            }
            classes[static_cast<std::size_t>(coordinate)] = static_cast<std::uint8_t>(class_id);
        }
        return std::make_pair(std::move(classes), std::move(representatives));
    };
    auto [y_classes, y_representatives] = build_band_classes(true);
    auto [x_classes, x_representatives] = build_band_classes(false);
    y_band_classes_ = std::move(y_classes);
    x_band_classes_ = std::move(x_classes);
    fast_nets_.resize(architecture.nets().size());
    for (std::size_t net_index = 0; net_index < architecture.nets().size(); ++net_index) {
        const auto& net = architecture.nets()[net_index];
        auto& fast = fast_nets_[net_index];
        fast.target_port = net.to;
        fast.horizontal = net.dx != 0;
        fast.axis_extent = static_cast<std::uint16_t>(
            fast.horizontal ? architecture.width() : architecture.height());
        const auto& representatives = fast.horizontal ? y_representatives : x_representatives;
        fast.band_count = static_cast<std::uint8_t>(representatives.size());
        fast.transitions.assign(static_cast<std::size_t>(fast.axis_extent) * fast.band_count, 0);
        for (std::size_t band = 0; band < representatives.size(); ++band) {
            for (int axis = 0; axis < fast.axis_extent; ++axis) {
                const int x = fast.horizontal ? axis : representatives[band];
                const int y = fast.horizontal ? representatives[band] : axis;
                const auto transition = architecture.resolve_net(x, y, net.from);
                if (!transition) continue;
                const auto target_axis = fast.horizontal ? transition->x : transition->y;
                fast.transitions[band * fast.axis_extent + static_cast<std::size_t>(axis)] =
                    (static_cast<std::uint32_t>(transition->delay) << 16) |
                    (static_cast<std::uint16_t>(target_axis) + 1u);
            }
        }
    }
    final_delays_.assign(architecture.ports().size(), std::numeric_limits<std::uint16_t>::max());
    std::vector<std::pair<int, int>> horizontal;
    std::vector<std::pair<int, int>> vertical;
    for (std::size_t from = 0; from < architecture.arcs().size(); ++from) {
        for (const auto& arc : architecture.arcs()[from]) {
            final_delays_[arc.to] = std::min(final_delays_[arc.to], arc.delay);
            auto& dense_delay = arc_delays_[from * port_count_ + arc.to];
            dense_delay = std::min(dense_delay, arc.delay);
            if (net_directions[arc.to] >= 0) {
                const auto& net = architecture.nets()[static_cast<std::size_t>(net_indices[arc.to])];
                routing_arcs_[from].push_back(RoutingArc{
                    static_cast<std::uint16_t>(net_indices[arc.to]), net.to, arc.delay,
                    net_directions[arc.to]});
            }
        }
    }
    for (const auto& net : architecture.nets()) {
        std::uint16_t cost = std::numeric_limits<std::uint16_t>::max();
        for (std::size_t input = 0; input < architecture.arcs().size(); ++input) {
            for (const auto& arc : architecture.arcs()[input]) {
                if (arc.to == net.from) cost = std::min(cost, arc.delay);
            }
        }
        if (cost == std::numeric_limits<std::uint16_t>::max()) continue;
        auto& moves = net.dx != 0 ? horizontal : vertical;
        const int delta = net.dx != 0 ? net.dx : net.dy;
        moves.emplace_back(delta, cost);
        for (const auto& block : architecture.block_gaps()) {
            const bool horizontal_move = net.dx != 0;
            const bool crossable = horizontal_move ? block.horizontal_crossable : block.vertical_crossable;
            if (!crossable) continue;
            const int size = horizontal_move ? block.right - block.left + 1 : block.upper - block.lower + 1;
            const int delay = horizontal_move ? block.horizontal_cross_delay : block.vertical_cross_delay;
            moves.emplace_back(delta + (delta > 0 ? size : -size), cost + delay);
        }
    }
    relaxed_x_ = relaxed_distances(architecture.width() - 1, horizontal);
    relaxed_y_ = relaxed_distances(architecture.height() - 1, vertical);
    vertical_gap_prefix_.assign(static_cast<std::size_t>(architecture.width()), 0);
    horizontal_gap_prefix_.assign(static_cast<std::size_t>(architecture.height()), 0);
    for (int x = 0; x < architecture.width(); ++x) {
        for (const auto& line : architecture.line_gaps()) if (!line.horizontal && line.site < x)
            vertical_gap_prefix_[static_cast<std::size_t>(x)] += line.delay;
    }
    for (int y = 0; y < architecture.height(); ++y) {
        for (const auto& line : architecture.line_gaps()) if (line.horizontal && line.site < y)
            horizontal_gap_prefix_[static_cast<std::size_t>(y)] += line.delay;
    }
}

const std::array<const char*, ShortFeatures::kFeatureCount>& ShortFeatures::names() {
    static constexpr std::array<const char*, kFeatureCount> result{{
        "a1_legal", "a1_blocked", "a1_duplicates", "a1_best", "a1_second", "a1_third",
        "a1_gap", "a1_near_best", "a1_directions", "a1_best_direction", "a1_toward_gap",
        "a1_best_reverse", "a1_best_overshoot", "a1_best_turn", "a1_topology_penalty",
        "a1_complete_cost", "a1_exact", "b2_legal", "b2_best", "b2_second", "b2_gap",
        "b2_complete_cost", "b2_exact", "b2_states",
        "b3_legal", "b3_best", "b3_second", "b3_gap", "b3_complete_cost", "b3_exact",
        "b3_states"
    }};
    return result;
}

std::int32_t ShortFeatures::heuristic(const State& state, const Endpoint& target) const {
    const int dx = target.x - state.x;
    const int dy = target.y - state.y;
    const auto x = relaxed_x_.at(static_cast<std::size_t>(dx + architecture_.width() - 1));
    const auto y = relaxed_y_.at(static_cast<std::size_t>(dy + architecture_.height() - 1));
    if (x > static_cast<std::uint32_t>(kInfinity) || y > static_cast<std::uint32_t>(kInfinity)) return kInfinity;
    const auto line_x = std::abs(static_cast<std::int64_t>(vertical_gap_prefix_[target.x]) - vertical_gap_prefix_[state.x]);
    const auto line_y = std::abs(static_cast<std::int64_t>(horizontal_gap_prefix_[target.y]) - horizontal_gap_prefix_[state.y]);
    const auto final = final_delays_[target.port] == std::numeric_limits<std::uint16_t>::max()
                           ? 0 : final_delays_[target.port];
    return static_cast<std::int32_t>(x + y + line_x + line_y + final);
}

std::int32_t ShortFeatures::complete_cost(const State& state, const Endpoint& target) const {
    if (state.x != target.x || state.y != target.y) return kMissing;
    const auto delay = arc_delays_[static_cast<std::size_t>(state.port) * port_count_ + target.port];
    return delay == std::numeric_limits<std::uint16_t>::max() ? kMissing : state.g + delay;
}

std::optional<NetTransition> ShortFeatures::resolve_arc(
    const State& state, const RoutingArc& arc) const {
    const auto& fast = fast_nets_[arc.net_index];
    const auto axis = static_cast<std::size_t>(fast.horizontal ? state.x : state.y);
    const auto band = static_cast<std::size_t>(
        fast.horizontal ? y_band_classes_[state.y] : x_band_classes_[state.x]);
    const auto packed = fast.transitions[band * fast.axis_extent + axis];
    if (packed == 0) return std::nullopt;
    const auto target_axis = static_cast<std::int16_t>((packed & 0xffffu) - 1u);
    return NetTransition{fast.horizontal ? target_axis : state.x,
                         fast.horizontal ? state.y : target_axis,
                         arc.target_port, static_cast<std::uint16_t>(packed >> 16)};
}

int ShortFeatures::expand_layer(const State* inputs, int input_count, const Endpoint& target,
                                std::array<State, kMaxStates>& output,
                                int& legal, int& blocked, int& duplicates) const {
    static constexpr int kHashSize = 256;
    std::array<std::int16_t, kHashSize> slots{};
    slots.fill(-1);
    int count = 0;
    for (int input_index = 0; input_index < input_count; ++input_index) {
        const auto& state = inputs[input_index];
        for (const auto& arc : routing_arcs_[state.port]) {
            const auto transition = resolve_arc(state, arc);
            if (!transition) { ++blocked; continue; }
            ++legal;
            State next{transition->x, transition->y, transition->to,
                       static_cast<std::int32_t>(state.g + arc.delay + transition->delay), 0,
                       arc.direction};
            next.estimate = next.g + heuristic(next, target);
            std::uint32_t hash = static_cast<std::uint32_t>(next.x) * 73856093u ^
                                 static_cast<std::uint32_t>(next.y) * 19349663u ^
                                 static_cast<std::uint32_t>(next.port) * 83492791u;
            int duplicate = -1;
            for (int probe = 0; probe < kHashSize; ++probe) {
                auto& slot = slots[(hash + static_cast<std::uint32_t>(probe)) & (kHashSize - 1)];
                if (slot < 0) {
                    if (count >= static_cast<int>(kMaxStates)) {
                        throw std::runtime_error("short feature state bound too small");
                    }
                    slot = static_cast<std::int16_t>(count);
                    output[count++] = next;
                    break;
                }
                const auto& known = output[slot];
                if (known.x == next.x && known.y == next.y && known.port == next.port) {
                    duplicate = slot;
                    break;
                }
            }
            if (duplicate >= 0) {
                ++duplicates;
                const auto old_key = std::tie(output[duplicate].g, output[duplicate].direction);
                const auto new_key = std::tie(next.g, next.direction);
                if (new_key < old_key) output[duplicate] = next;
            }
        }
    }
    return count;
}

void ShortFeatures::sort_states(std::array<State, kMaxStates>& states, int count) {
    std::sort(states.begin(), states.begin() + count, [](const State& left, const State& right) {
        return std::tie(left.estimate, left.g, left.x, left.y, left.port, left.direction) <
               std::tie(right.estimate, right.g, right.x, right.y, right.port, right.direction);
    });
}

std::array<std::int32_t, ShortFeatures::kFeatureCount> ShortFeatures::feature_for(
    const Endpoint& from, const Endpoint& to, int maximum_depth) const {
    if (maximum_depth < 2 || maximum_depth > 3) {
        throw std::runtime_error("short feature depth must be 2 or 3");
    }
    std::array<std::int32_t, kFeatureCount> values{};
    values.fill(kMissing);
    State start{from.x, from.y, from.port, 0, 0, -1};
    if (architecture_.ports()[from.port].direction == Direction::Output && architecture_.ports()[from.port].inter) {
        const auto source_net = architecture_.resolve_net(from.x, from.y, from.port);
        if (source_net) start = {source_net->x, source_net->y, source_net->to, source_net->delay, 0, -1};
    }
    start.estimate = start.g + heuristic(start, to);
    std::array<State, kMaxStates> first{};
    int legal = 0, blocked = 0, duplicates = 0;
    int first_count = expand_layer(&start, 1, to, first, legal, blocked, duplicates);
    sort_states(first, first_count);
    values[0] = legal; values[1] = blocked; values[2] = duplicates;
    if (first_count == 0) return values;
    values[3] = first[0].estimate;
    values[4] = first_count > 1 ? first[1].estimate : kMissing;
    values[5] = first_count > 2 ? first[2].estimate : kMissing;
    values[6] = first_count > 1 ? first[1].estimate - first[0].estimate : kMissing;
    int near = 0; bool directions[4]{};
    for (int index = 0; index < first_count; ++index) {
        if (first[index].estimate <= first[0].estimate + 50) ++near;
        if (first[index].direction >= 0) directions[first[index].direction] = true;
    }
    values[7] = near; values[8] = int(directions[0]) + int(directions[1]) + int(directions[2]) + int(directions[3]);
    values[9] = first[0].direction;
    const int dx = to.x - start.x, dy = to.y - start.y;
    const int toward = dx > 0 ? 0 : dx < 0 ? 1 : dy > 0 ? 2 : dy < 0 ? 3 : -1;
    int toward_best = kInfinity;
    for (int index = 0; index < first_count; ++index) if (first[index].direction == toward)
        toward_best = std::min(toward_best, first[index].estimate);
    values[10] = toward_best == kInfinity ? kMissing : toward_best - first[0].estimate;
    values[11] = (toward >= 0 && first[0].direction != toward) ? 1 : 0;
    const int before = std::abs(dx) + std::abs(dy);
    const int after = std::abs(to.x - first[0].x) + std::abs(to.y - first[0].y);
    values[12] = after > before ? 1 : 0;
    values[13] = (first[0].direction >= 0 && toward >= 0 && first[0].direction != toward) ? 1 : 0;
    values[14] = first[0].estimate - start.estimate;
    int best_complete = complete_cost(start, to);
    for (int index = 0; index < first_count; ++index) {
        const int complete = complete_cost(first[index], to);
        if (complete >= 0 && (best_complete < 0 || complete < best_complete)) best_complete = complete;
    }
    values[15] = best_complete; values[16] = best_complete >= 0 ? 1 : 0;

    std::array<State, kMaxStates> second{};
    int b2_legal = 0, ignored_blocked = 0, ignored_duplicates = 0;
    const int second_count = expand_layer(first.data(), std::min(first_count, 4), to, second,
                                          b2_legal, ignored_blocked, ignored_duplicates);
    sort_states(second, second_count);
    values[17] = b2_legal; values[23] = second_count;
    if (second_count > 0) {
        values[18] = second[0].estimate;
        values[19] = second_count > 1 ? second[1].estimate : kMissing;
        values[20] = second_count > 1 ? second[1].estimate - second[0].estimate : kMissing;
        for (int index = 0; index < second_count; ++index) {
            const int complete = complete_cost(second[index], to);
            if (complete >= 0 && (best_complete < 0 || complete < best_complete)) best_complete = complete;
        }
    }
    values[21] = best_complete; values[22] = best_complete >= 0 ? 1 : 0;
    if (maximum_depth == 2) return values;

    std::array<State, kMaxStates> third{};
    int b3_legal = 0;
    const int third_count = expand_layer(second.data(), std::min(second_count, 2), to, third,
                                         b3_legal, ignored_blocked, ignored_duplicates);
    sort_states(third, third_count);
    values[24] = b3_legal; values[30] = third_count;
    if (third_count > 0) {
        values[25] = third[0].estimate;
        values[26] = third_count > 1 ? third[1].estimate : kMissing;
        values[27] = third_count > 1 ? third[1].estimate - third[0].estimate : kMissing;
        for (int index = 0; index < third_count; ++index) {
            const int complete = complete_cost(third[index], to);
            if (complete >= 0 && (best_complete < 0 || complete < best_complete)) best_complete = complete;
        }
    }
    values[28] = best_complete; values[29] = best_complete >= 0 ? 1 : 0;
    return values;
}

}  // namespace delay
