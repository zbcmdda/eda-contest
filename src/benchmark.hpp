#pragma once

#include "architecture.hpp"
#include "ml_estimator.hpp"
#include "router.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace delay {

struct BenchmarkSummary {
    std::size_t rows = 0;
    std::size_t mismatches = 0;
    std::uint64_t total_popped = 0;
    std::uint64_t total_expanded_arcs = 0;
    double elapsed_seconds = 0.0;
};

struct ModelBenchmarkSummary {
    std::size_t rows = 0;
    double official_score = 0.0;
    double mean_relative_error = 0.0;
    double within_5_percent = 0.0;
    double within_10_percent = 0.0;
    double within_20_percent = 0.0;
    double elapsed_seconds = 0.0;
};

[[nodiscard]] BenchmarkSummary benchmark_answers(
    const Architecture& architecture,
    Router& router,
    const std::string& path,
    std::size_t limit,
    std::size_t stride);

[[nodiscard]] ModelBenchmarkSummary benchmark_model(
    const Architecture& architecture,
    const MlEstimator& estimator,
    const std::string& path,
    std::size_t limit,
    std::size_t stride,
    std::size_t offset);

}  // namespace delay
