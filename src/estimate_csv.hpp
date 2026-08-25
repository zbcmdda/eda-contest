#pragma once

#include "architecture.hpp"
#include "ml_estimator.hpp"
#include "router.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace delay {

struct EstimateCsvSummary {
    std::size_t rows = 0;
    std::size_t exact_rows = 0;
    std::size_t model_rows = 0;
    std::size_t exact_unreachable_rows = 0;
    std::uint64_t route_popped = 0;
    std::uint64_t route_expanded_arcs = 0;
    double elapsed_seconds = 0.0;
};

[[nodiscard]] EstimateCsvSummary estimate_csv(
    const Architecture& architecture,
    const MlEstimator& estimator,
    const std::string& input_path,
    const std::string& output_path,
    Router* exact_router = nullptr,
    int exact_manhattan_threshold = -1,
    std::uint32_t route_heuristic_weight_milli = 1000,
    bool calibrate_short_route = false);

}  // namespace delay
