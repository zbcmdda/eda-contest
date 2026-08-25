#pragma once

#include "architecture.hpp"
#include "router.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace delay {

// Offline-only, bounded local routing descriptors for short-path experiments.
// No heap allocation or priority queue occurs in feature_for(); depth two keeps
// the four best first-hop states with a stable tie break.
class ShortFeatures {
public:
    static constexpr std::size_t kFeatureCount = 31;
    explicit ShortFeatures(const Architecture& architecture);
    [[nodiscard]] std::array<std::int32_t, kFeatureCount> feature_for(
        const Endpoint& from, const Endpoint& to, int maximum_depth = 3) const;
    [[nodiscard]] static const std::array<const char*, kFeatureCount>& names();

private:
    struct State {
        std::int16_t x = 0;
        std::int16_t y = 0;
        std::uint16_t port = 0;
        std::int32_t g = 0;
        std::int32_t estimate = 0;
        std::int8_t direction = -1;
    };
    struct RoutingArc {
        std::uint16_t net_index = 0;
        std::uint16_t target_port = 0;
        std::uint16_t delay = 0;
        std::int8_t direction = -1;
    };
    struct FastNet {
        std::vector<std::uint32_t> transitions;
        std::uint16_t target_port = 0;
        std::uint16_t axis_extent = 0;
        std::uint8_t band_count = 0;
        bool horizontal = false;
    };
    static constexpr std::size_t kMaxStates = 128;

    [[nodiscard]] std::int32_t heuristic(const State& state, const Endpoint& target) const;
    [[nodiscard]] std::int32_t complete_cost(const State& state, const Endpoint& target) const;
    [[nodiscard]] std::optional<NetTransition> resolve_arc(
        const State& state, const RoutingArc& arc) const;
    [[nodiscard]] int expand_layer(const State* inputs, int input_count, const Endpoint& target,
                                   std::array<State, kMaxStates>& output,
                                   int& legal, int& blocked, int& duplicates) const;
    static void sort_states(std::array<State, kMaxStates>& states, int count);

    const Architecture& architecture_;
    std::size_t port_count_ = 0;
    std::vector<std::vector<RoutingArc>> routing_arcs_;
    std::vector<FastNet> fast_nets_;
    std::vector<std::uint8_t> x_band_classes_;
    std::vector<std::uint8_t> y_band_classes_;
    std::vector<std::uint16_t> arc_delays_;
    std::vector<std::uint16_t> final_delays_;
    std::vector<std::uint32_t> vertical_gap_prefix_;
    std::vector<std::uint32_t> horizontal_gap_prefix_;
    std::vector<std::uint32_t> relaxed_x_;
    std::vector<std::uint32_t> relaxed_y_;
};

}  // namespace delay
