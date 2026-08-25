#pragma once

#include "architecture.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace delay {

struct Endpoint {
    std::int16_t x;
    std::int16_t y;
    std::uint16_t port;

    friend bool operator==(const Endpoint& left, const Endpoint& right) {
        return left.x == right.x && left.y == right.y && left.port == right.port;
    }
};

struct SearchStats {
    std::uint64_t popped = 0;
    std::uint64_t expanded_arcs = 0;
};

[[nodiscard]] Endpoint parse_endpoint(const Architecture& architecture, std::string_view text);

class Router {
public:
    explicit Router(const Architecture& architecture);

    [[nodiscard]] std::optional<std::uint32_t> shortest_delay(
        const Endpoint& from, const Endpoint& to, SearchStats* stats = nullptr);

    // Weighted A* keeps the same routing graph but prioritizes the admissible
    // heuristic more aggressively. 1000 is bit-for-bit equivalent to exact A*;
    // larger values trade a bounded amount of path optimality for fewer states.
    [[nodiscard]] std::optional<std::uint32_t> shortest_delay_weighted(
        const Endpoint& from, const Endpoint& to, std::uint32_t heuristic_weight_milli,
        SearchStats* stats = nullptr);

private:
    struct RoutingArc {
        std::uint16_t net_index = 0;
        std::uint16_t target_port = 0;
        std::uint16_t delay = 0;
    };
    struct FastNet {
        std::vector<std::uint32_t> transitions;
        std::uint16_t axis_extent = 0;
        std::uint8_t band_count = 0;
        bool horizontal = false;
    };

    [[nodiscard]] std::uint32_t heuristic(
        int x, int y, std::uint16_t input_port, const Endpoint& target) const;
    [[nodiscard]] std::optional<NetTransition> resolve_routing_arc(
        int x, int y, const RoutingArc& arc) const;

    [[nodiscard]] std::optional<std::uint32_t> shortest_delay_impl(
        const Endpoint& from, const Endpoint& to, std::uint32_t heuristic_weight_milli,
        SearchStats* stats);

    const Architecture& architecture_;
    std::vector<std::int16_t> input_indices_;
    std::vector<std::uint16_t> input_ports_;
    std::vector<std::uint32_t> distances_;
    std::vector<std::uint32_t> generations_;
    std::vector<std::uint16_t> first_routing_delays_;
    std::vector<std::uint16_t> final_delays_;
    std::vector<std::vector<RoutingArc>> routing_arcs_;
    std::vector<FastNet> fast_nets_;
    std::vector<std::uint8_t> x_band_classes_;
    std::vector<std::uint8_t> y_band_classes_;
    std::vector<std::uint16_t> direct_arc_delays_;
    std::vector<std::uint32_t> port_lower_bounds_;
    std::vector<std::uint32_t> vertical_gap_prefix_;
    std::vector<std::uint32_t> horizontal_gap_prefix_;
    std::vector<std::uint32_t> horizontal_relaxed_distances_;
    std::vector<std::uint32_t> vertical_relaxed_distances_;
    int horizontal_relaxed_extent_ = 0;
    int vertical_relaxed_extent_ = 0;
    std::uint16_t minimum_routing_delay_ = 0;
    std::uint16_t maximum_horizontal_span_ = 1;
    std::uint16_t maximum_vertical_span_ = 1;
    std::uint32_t generation_ = 0;
};

}  // namespace delay
