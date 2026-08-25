#pragma once

#include "architecture.hpp"
#include "router.hpp"
#include "short_features.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace delay {

class MlEstimator {
public:
    explicit MlEstimator(const Architecture& architecture, std::size_t tree_limit = 128,
                         bool short_residual = false, int short_residual_threshold = 128);

    [[nodiscard]] std::uint32_t estimate(const Endpoint& from, const Endpoint& to) const;

private:
    [[nodiscard]] std::array<double, 51> features(
        const Endpoint& from, const Endpoint& to,
        std::uint16_t* effective_source_port = nullptr) const;

    const Architecture& architecture_;
    std::vector<std::uint16_t> final_delays_;
    std::vector<std::uint16_t> first_routing_delays_;
    std::vector<std::uint32_t> vertical_gap_prefix_;
    std::vector<std::uint32_t> horizontal_gap_prefix_;
    std::vector<std::uint32_t> relaxed_x_;
    std::vector<std::uint32_t> relaxed_y_;
    // These values depend only on a port id.  Keeping them out of the query
    // hot path avoids repeatedly decoding the short port-name strings and
    // binary-searching the generated categorical dictionaries.
    std::vector<std::array<std::int32_t, 9>> port_metadata_;
    std::vector<std::int32_t> endpoint_pair_categories_;
    std::vector<std::array<std::int32_t, 3>> medium_short_port_categories_;
    std::vector<std::array<std::int32_t, 3>> ultrashort_port_categories_;
    std::vector<std::int32_t> medium_short_endpoint_pair_categories_;
    std::vector<std::int32_t> ultrashort_endpoint_pair_categories_;
    std::unique_ptr<ShortFeatures> short_features_;
    std::size_t tree_limit_;
    int short_residual_threshold_ = 128;
};

}  // namespace delay
