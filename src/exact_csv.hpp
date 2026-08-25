#pragma once

#include "architecture.hpp"
#include "router.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace delay {

struct ExactCsvSummary {
    std::size_t input_rows = 0;
    std::size_t labeled_rows = 0;
    std::size_t unreachable_rows = 0;
    std::uint64_t total_popped = 0;
    std::uint64_t total_expanded_arcs = 0;
    double elapsed_seconds = 0.0;
};

[[nodiscard]] ExactCsvSummary label_exact_csv(
    const Architecture& architecture,
    Router& router,
    const std::string& input_path,
    const std::string& output_path,
    std::size_t offset,
    std::size_t limit);

}  // namespace delay
