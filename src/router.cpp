#include "router.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace delay {
namespace {

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' || text.front() == '\r')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    return text;
}

int parse_coordinate(std::string_view text, const char* field) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size() || value < 0) {
        throw std::runtime_error(std::string("invalid ") + field + " coordinate");
    }
    return value;
}

struct RelaxedMove {
    int delta;
    std::uint16_t cost;
};

struct QueueItem {
    std::uint64_t priority;
    std::uint32_t distance;
    std::uint32_t state;

    bool operator>(const QueueItem& other) const {
        return priority > other.priority;
    }
};

std::vector<std::uint32_t> build_relaxed_distances(
    int maximum_query_delta, const std::vector<RelaxedMove>& moves, int& extent) {
    int maximum_move = 1;
    for (const auto& move : moves) {
        maximum_move = std::max(maximum_move, std::abs(move.delta));
    }
    extent = maximum_query_delta + maximum_move;
    const auto infinite = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> distances(static_cast<std::size_t>(extent * 2 + 1), infinite);
    using Item = std::pair<std::uint32_t, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    distances[extent] = 0;
    queue.emplace(0, 0);
    while (!queue.empty()) {
        const auto [distance, coordinate] = queue.top();
        queue.pop();
        if (distance != distances[static_cast<std::size_t>(coordinate + extent)]) {
            continue;
        }
        for (const auto& move : moves) {
            const int next = coordinate + move.delta;
            if (next < -extent || next > extent) {
                continue;
            }
            const std::uint32_t candidate = distance + move.cost;
            auto& current = distances[static_cast<std::size_t>(next + extent)];
            if (candidate < current) {
                current = candidate;
                queue.emplace(candidate, next);
            }
        }
    }
    return distances;
}

}  // namespace

Endpoint parse_endpoint(const Architecture& architecture, std::string_view raw_text) {
    const auto text = trim(raw_text);
    constexpr std::string_view prefix = "SRB_";
    if (text.substr(0, prefix.size()) != prefix) {
        throw std::runtime_error("endpoint does not start with SRB_: " + std::string(text));
    }
    const auto separator = text.find('/');
    const auto underscore = text.find('_', prefix.size());
    if (separator == std::string_view::npos || underscore == std::string_view::npos ||
        underscore >= separator || separator + 1 == text.size()) {
        throw std::runtime_error("malformed endpoint: " + std::string(text));
    }
    const int x = parse_coordinate(text.substr(prefix.size(), underscore - prefix.size()), "x");
    const int y = parse_coordinate(text.substr(underscore + 1, separator - underscore - 1), "y");
    if (!architecture.active(x, y)) {
        throw std::runtime_error("endpoint references missing SRB: " + std::string(text));
    }
    if (x > std::numeric_limits<std::int16_t>::max() || y > std::numeric_limits<std::int16_t>::max()) {
        throw std::runtime_error("endpoint coordinate exceeds 16-bit range");
    }
    return Endpoint{
        static_cast<std::int16_t>(x),
        static_cast<std::int16_t>(y),
        architecture.port_id(text.substr(separator + 1)),
    };
}

Router::Router(const Architecture& architecture) : architecture_(architecture) {
    input_indices_.assign(architecture.ports().size(), -1);
    for (std::size_t port = 0; port < architecture.ports().size(); ++port) {
        if (architecture.ports()[port].direction == Direction::Input) {
            if (input_ports_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int16_t>::max())) {
                throw std::runtime_error("too many input ports");
            }
            input_indices_[port] = static_cast<std::int16_t>(input_ports_.size());
            input_ports_.push_back(static_cast<std::uint16_t>(port));
        }
    }
    const std::size_t state_count = static_cast<std::size_t>(architecture.width()) *
                                    architecture.height() * input_ports_.size();
    if (state_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("routing state space exceeds 32-bit index");
    }
    distances_.resize(state_count);
    generations_.assign(state_count, 0);

    const auto infinite = std::numeric_limits<std::uint16_t>::max();
    first_routing_delays_.assign(architecture.ports().size(), infinite);
    final_delays_.assign(architecture.ports().size(), infinite);
    routing_arcs_.resize(architecture.ports().size());
    direct_arc_delays_.assign(
        architecture.ports().size() * architecture.ports().size(), infinite);
    minimum_routing_delay_ = infinite;
    std::vector<std::int16_t> net_indices(architecture.ports().size(), -1);
    for (std::size_t index = 0; index < architecture.nets().size(); ++index) {
        net_indices[architecture.nets()[index].from] = static_cast<std::int16_t>(index);
    }
    std::vector<std::uint16_t> minimum_net_arc(architecture.nets().size(), infinite);
    for (std::size_t from = 0; from < architecture.arcs().size(); ++from) {
        for (const auto& arc : architecture.arcs()[from]) {
            auto& direct = direct_arc_delays_[from * architecture.ports().size() + arc.to];
            direct = std::min(direct, arc.delay);
            final_delays_[arc.to] = std::min(final_delays_[arc.to], arc.delay);
            if (architecture.ports()[arc.to].inter) {
                const auto net_index = net_indices[arc.to];
                if (net_index < 0) {
                    throw std::runtime_error("inter-port Arc has no Net");
                }
                routing_arcs_[from].push_back(RoutingArc{
                    static_cast<std::uint16_t>(net_index),
                    architecture.nets()[static_cast<std::size_t>(net_index)].to,
                    arc.delay});
                first_routing_delays_[from] = std::min(first_routing_delays_[from], arc.delay);
                minimum_routing_delay_ = std::min(minimum_routing_delay_, arc.delay);
                for (std::size_t net_index = 0; net_index < architecture.nets().size(); ++net_index) {
                    if (architecture.nets()[net_index].from == arc.to) {
                        minimum_net_arc[net_index] = std::min(minimum_net_arc[net_index], arc.delay);
                        break;
                    }
                }
            }
        }
    }
    if (minimum_routing_delay_ == infinite) {
        throw std::runtime_error("architecture has no routable Arc");
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
                if (inside) {
                    mask |= static_cast<std::uint8_t>(1u << index);
                }
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
                if (!transition) {
                    continue;
                }
                const auto target_axis = fast.horizontal ? transition->x : transition->y;
                fast.transitions[band * fast.axis_extent + static_cast<std::size_t>(axis)] =
                    (static_cast<std::uint32_t>(transition->delay) << 16) |
                    (static_cast<std::uint16_t>(target_axis) + 1u);
            }
        }
    }

    // Exact all-pairs distances on the coordinate-free input-port graph. Any
    // physical route projects onto this graph, so these costs are admissible
    // lower bounds while retaining port compatibility that the spatial
    // relaxed heuristic intentionally ignores.
    const auto port_count = architecture.ports().size();
    const auto input_count = input_ports_.size();
    const auto unreachable = std::numeric_limits<std::uint32_t>::max() / 4;
    std::vector<std::uint32_t> abstract_distances(input_count * input_count, unreachable);
    for (std::size_t input = 0; input < input_count; ++input) {
        abstract_distances[input * input_count + input] = 0;
        const auto from_port = input_ports_[input];
        for (const auto& arc : routing_arcs_[from_port]) {
            const auto target_index = input_indices_[arc.target_port];
            if (target_index < 0) {
                throw std::runtime_error("Net target is not an input port");
            }
            auto& current = abstract_distances[
                input * input_count + static_cast<std::size_t>(target_index)];
            current = std::min<std::uint32_t>(current, arc.delay);
        }
    }
    for (std::size_t middle = 0; middle < input_count; ++middle) {
        for (std::size_t from = 0; from < input_count; ++from) {
            const auto first = abstract_distances[from * input_count + middle];
            if (first == unreachable) {
                continue;
            }
            for (std::size_t to = 0; to < input_count; ++to) {
                const auto second = abstract_distances[middle * input_count + to];
                if (second == unreachable) {
                    continue;
                }
                auto& current = abstract_distances[from * input_count + to];
                current = std::min(current, first + second);
            }
        }
    }
    port_lower_bounds_.assign(port_count * port_count, unreachable);
    for (std::size_t from = 0; from < input_count; ++from) {
        const auto from_port = input_ports_[from];
        for (std::size_t target = 0; target < port_count; ++target) {
            auto& result = port_lower_bounds_[from_port * port_count + target];
            const auto target_input = input_indices_[target];
            if (target_input >= 0) {
                result = abstract_distances[
                    from * input_count + static_cast<std::size_t>(target_input)];
            }
            for (std::size_t terminal = 0; terminal < input_count; ++terminal) {
                const auto prefix = abstract_distances[from * input_count + terminal];
                const auto direct = direct_arc_delays_[input_ports_[terminal] * port_count + target];
                if (prefix != unreachable && direct != infinite) {
                    result = std::min(result, prefix + direct);
                }
            }
        }
    }

    int maximum_horizontal_block_width = 0;
    int maximum_vertical_block_height = 0;
    for (int y = 0; y < architecture.height(); ++y) {
        int total = 0;
        for (const auto& block : architecture.block_gaps()) {
            if (block.horizontal_crossable && y >= block.lower && y <= block.upper) {
                total += block.right - block.left + 1;
            }
        }
        maximum_horizontal_block_width = std::max(maximum_horizontal_block_width, total);
    }
    for (int x = 0; x < architecture.width(); ++x) {
        int total = 0;
        for (const auto& block : architecture.block_gaps()) {
            if (block.vertical_crossable && x >= block.left && x <= block.right) {
                total += block.upper - block.lower + 1;
            }
        }
        maximum_vertical_block_height = std::max(maximum_vertical_block_height, total);
    }
    int maximum_net_x = 0;
    int maximum_net_y = 0;
    for (const auto& net : architecture.nets()) {
        maximum_net_x = std::max(maximum_net_x, std::abs(static_cast<int>(net.dx)));
        maximum_net_y = std::max(maximum_net_y, std::abs(static_cast<int>(net.dy)));
    }
    maximum_horizontal_span_ = static_cast<std::uint16_t>(maximum_net_x + maximum_horizontal_block_width);
    maximum_vertical_span_ = static_cast<std::uint16_t>(maximum_net_y + maximum_vertical_block_height);

    std::vector<RelaxedMove> horizontal_moves;
    std::vector<RelaxedMove> vertical_moves;
    for (std::size_t net_index = 0; net_index < architecture.nets().size(); ++net_index) {
        const auto& net = architecture.nets()[net_index];
        const auto cost = minimum_net_arc[net_index];
        if (cost == infinite) {
            continue;
        }
        auto& moves = net.dx != 0 ? horizontal_moves : vertical_moves;
        const int base_delta = net.dx != 0 ? net.dx : net.dy;
        moves.push_back(RelaxedMove{base_delta, cost});
        for (const auto& block : architecture.block_gaps()) {
            const bool crossable = net.dx != 0 ? block.horizontal_crossable : block.vertical_crossable;
            if (!crossable) {
                continue;
            }
            const int block_size = net.dx != 0 ? block.right - block.left + 1
                                               : block.upper - block.lower + 1;
            const auto block_delay = net.dx != 0 ? block.horizontal_cross_delay
                                                 : block.vertical_cross_delay;
            const int extended_delta = base_delta + (base_delta > 0 ? block_size : -block_size);
            moves.push_back(RelaxedMove{
                extended_delta,
                static_cast<std::uint16_t>(cost + block_delay),
            });
        }
    }
    horizontal_relaxed_distances_ = build_relaxed_distances(
        architecture.width() - 1, horizontal_moves, horizontal_relaxed_extent_);
    vertical_relaxed_distances_ = build_relaxed_distances(
        architecture.height() - 1, vertical_moves, vertical_relaxed_extent_);

    vertical_gap_prefix_.resize(architecture.width());
    for (int x = 0; x < architecture.width(); ++x) {
        std::uint32_t total = 0;
        for (const auto& line : architecture.line_gaps()) {
            if (!line.horizontal && line.site < x) {
                total += line.delay;
            }
        }
        vertical_gap_prefix_[x] = total;
    }
    horizontal_gap_prefix_.resize(architecture.height());
    for (int y = 0; y < architecture.height(); ++y) {
        std::uint32_t total = 0;
        for (const auto& line : architecture.line_gaps()) {
            if (line.horizontal && line.site < y) {
                total += line.delay;
            }
        }
        horizontal_gap_prefix_[y] = total;
    }
}

std::uint32_t Router::heuristic(
    int x, int y, std::uint16_t input_port, const Endpoint& target) const {
    const auto absolute_x = static_cast<std::uint32_t>(std::abs(x - target.x));
    const auto absolute_y = static_cast<std::uint32_t>(std::abs(y - target.y));
    const auto horizontal_hops = (absolute_x + maximum_horizontal_span_ - 1) / maximum_horizontal_span_;
    const auto vertical_hops = (absolute_y + maximum_vertical_span_ - 1) / maximum_vertical_span_;
    const auto routing_hops = horizontal_hops + vertical_hops;

    std::uint32_t hop_result = 0;
    if (routing_hops != 0) {
        const auto first = first_routing_delays_.at(input_port);
        if (first == std::numeric_limits<std::uint16_t>::max()) {
            return std::numeric_limits<std::uint32_t>::max() / 2;
        }
        hop_result += first;
        hop_result += (routing_hops - 1) * minimum_routing_delay_;
    }
    std::uint32_t terminal_delay = 0;
    if (architecture_.ports()[target.port].direction == Direction::Output) {
        const auto final = final_delays_[target.port];
        if (final == std::numeric_limits<std::uint16_t>::max()) {
            return std::numeric_limits<std::uint32_t>::max() / 2;
        }
        terminal_delay = final;
    }
    const auto vertical_low = std::min<std::uint32_t>(vertical_gap_prefix_[x], vertical_gap_prefix_[target.x]);
    const auto vertical_high = std::max<std::uint32_t>(vertical_gap_prefix_[x], vertical_gap_prefix_[target.x]);
    const auto horizontal_low = std::min<std::uint32_t>(horizontal_gap_prefix_[y], horizontal_gap_prefix_[target.y]);
    const auto horizontal_high = std::max<std::uint32_t>(horizontal_gap_prefix_[y], horizontal_gap_prefix_[target.y]);
    const std::uint32_t gap_delay = vertical_high - vertical_low + horizontal_high - horizontal_low;
    const int delta_x = target.x - x;
    const int delta_y = target.y - y;
    const auto relaxed_x = horizontal_relaxed_distances_.at(
        static_cast<std::size_t>(delta_x + horizontal_relaxed_extent_));
    const auto relaxed_y = vertical_relaxed_distances_.at(
        static_cast<std::size_t>(delta_y + vertical_relaxed_extent_));
    const std::uint32_t relaxed_result = relaxed_x + relaxed_y;
    const auto spatial_result = std::max(hop_result, relaxed_result) + terminal_delay;
    const auto port_result = port_lower_bounds_[
        static_cast<std::size_t>(input_port) * architecture_.ports().size() + target.port];
    return std::max(spatial_result, port_result) + gap_delay;
}

std::optional<NetTransition> Router::resolve_routing_arc(
    int x, int y, const RoutingArc& arc) const {
    const auto& fast = fast_nets_[arc.net_index];
    const auto axis = static_cast<std::size_t>(fast.horizontal ? x : y);
    const auto band = static_cast<std::size_t>(
        fast.horizontal ? y_band_classes_[y] : x_band_classes_[x]);
    const auto packed = fast.transitions[band * fast.axis_extent + axis];
    if (packed == 0) {
        return std::nullopt;
    }
    const auto target_axis = static_cast<std::int16_t>((packed & 0xffffu) - 1u);
    return NetTransition{
        fast.horizontal ? target_axis : static_cast<std::int16_t>(x),
        fast.horizontal ? static_cast<std::int16_t>(y) : target_axis,
        arc.target_port,
        static_cast<std::uint16_t>(packed >> 16)};
}

std::optional<std::uint32_t> Router::shortest_delay(
    const Endpoint& from, const Endpoint& to, SearchStats* stats) {
    return shortest_delay_impl(from, to, 1000, stats);
}

std::optional<std::uint32_t> Router::shortest_delay_weighted(
    const Endpoint& from, const Endpoint& to, std::uint32_t heuristic_weight_milli,
    SearchStats* stats) {
    if (heuristic_weight_milli < 1000 || heuristic_weight_milli > 10000) {
        throw std::invalid_argument("weighted A* heuristic must be between 1000 and 10000 milli");
    }
    return shortest_delay_impl(from, to, heuristic_weight_milli, stats);
}

std::optional<std::uint32_t> Router::shortest_delay_impl(
    const Endpoint& from, const Endpoint& to, std::uint32_t heuristic_weight_milli,
    SearchStats* stats) {
    if (stats != nullptr) {
        *stats = {};
    }
    if (from == to) {
        return 0;
    }
    if (++generation_ == 0) {
        std::fill(generations_.begin(), generations_.end(), 0);
        generation_ = 1;
    }
    const auto input_count = static_cast<std::uint32_t>(input_ports_.size());
    auto state_for = [&](int x, int y, std::uint16_t port) -> std::optional<std::uint32_t> {
        if (port >= input_indices_.size() || input_indices_[port] < 0) {
            return std::nullopt;
        }
        const auto site = static_cast<std::uint32_t>(y * architecture_.width() + x);
        return site * input_count + static_cast<std::uint16_t>(input_indices_[port]);
    };

    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;
    auto relax = [&](std::uint32_t state, std::uint32_t distance) {
        if (generations_[state] != generation_ || distance < distances_[state]) {
            generations_[state] = generation_;
            distances_[state] = distance;
            const auto site = state / input_count;
            const int x = static_cast<int>(site % architecture_.width());
            const int y = static_cast<int>(site / architecture_.width());
            const auto input_port = input_ports_[state % input_count];
            const auto lower_bound = heuristic(x, y, input_port, to);
            const auto weighted = static_cast<std::uint64_t>(lower_bound) *
                                  heuristic_weight_milli / 1000;
            queue.push(QueueItem{distance + weighted, distance, state});
        }
    };

    const auto& from_port = architecture_.ports().at(from.port);
    if (from_port.direction == Direction::Input) {
        relax(*state_for(from.x, from.y, from.port), 0);
    } else if (from_port.inter) {
        const auto transition = architecture_.resolve_net(from.x, from.y, from.port);
        if (!transition) {
            return std::nullopt;
        }
        relax(*state_for(transition->x, transition->y, transition->to), transition->delay);
    } else {
        return std::nullopt;
    }

    std::uint32_t best = std::numeric_limits<std::uint32_t>::max();
    while (!queue.empty()) {
        const auto item = queue.top();
        queue.pop();
        if (item.priority >= best) {
            break;
        }
        if (generations_[item.state] != generation_ || distances_[item.state] != item.distance) {
            continue;
        }
        if (stats != nullptr) {
            ++stats->popped;
        }
        const auto site = item.state / input_count;
        const int x = static_cast<int>(site % architecture_.width());
        const int y = static_cast<int>(site / architecture_.width());
        const auto input_port = input_ports_[item.state % input_count];
        if (x == to.x && y == to.y && input_port == to.port) {
            best = item.distance;
            break;
        }

        if (x == to.x && y == to.y) {
            const auto direct = direct_arc_delays_[
                static_cast<std::size_t>(input_port) * architecture_.ports().size() + to.port];
            if (direct != std::numeric_limits<std::uint16_t>::max()) {
                best = std::min(best, item.distance + direct);
            }
        }
        for (const auto& arc : routing_arcs_[input_port]) {
            if (stats != nullptr) {
                ++stats->expanded_arcs;
            }
            const std::uint32_t arc_distance = item.distance + arc.delay;
            const auto transition = resolve_routing_arc(x, y, arc);
            if (!transition) {
                continue;
            }
            const auto next_state = state_for(transition->x, transition->y, transition->to);
            if (!next_state) {
                throw std::runtime_error("Net target is not an input port");
            }
            relax(*next_state, arc_distance + transition->delay);
        }
    }
    if (best == std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return best;
}

}  // namespace delay
