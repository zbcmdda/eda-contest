#pragma once

#include "architecture.hpp"
#include "router.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace delay {

struct CheckSummary {
    std::size_t rows = 0;
    std::uint64_t total_popped = 0;
    std::uint64_t total_expanded_arcs = 0;
};

[[nodiscard]] CheckSummary verify_check_file(
    const Architecture& architecture, Router& router, const std::string& path);

}  // namespace delay
