#include "estimate_csv.hpp"

#include "router.hpp"

#include <chrono>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
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

}  // namespace

EstimateCsvSummary estimate_csv(
    const Architecture& architecture,
    const MlEstimator& estimator,
    const std::string& input_path,
    const std::string& output_path,
    Router* exact_router,
    int exact_manhattan_threshold,
    std::uint32_t route_heuristic_weight_milli,
    bool calibrate_short_route) {
    if ((exact_router == nullptr) != (exact_manhattan_threshold < 0)) {
        throw std::runtime_error("exact router and Manhattan threshold must be enabled together");
    }
    if (std::filesystem::absolute(input_path).lexically_normal() ==
        std::filesystem::absolute(output_path).lexically_normal()) {
        throw std::runtime_error("input and output CSV paths must differ");
    }
    std::ifstream input;
    std::vector<char> input_buffer(1 << 20);
    input.rdbuf()->pubsetbuf(input_buffer.data(), static_cast<std::streamsize>(input_buffer.size()));
    input.open(input_path);
    if (!input) {
        throw std::runtime_error("cannot open request CSV: " + input_path);
    }
    std::string line;
    if (!std::getline(input, line) || trim(line) != "From,To") {
        throw std::runtime_error("unexpected request CSV header");
    }

    std::ofstream output;
    std::vector<char> output_buffer(1 << 20);
    output.rdbuf()->pubsetbuf(output_buffer.data(), static_cast<std::streamsize>(output_buffer.size()));
    output.open(output_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open result CSV: " + output_path);
    }
    output << "From,To,delay\n";

    EstimateCsvSummary summary;
    // Keep formatting out of std::ostream's locale-aware integer inserter.
    // The request rows are short and fixed-shape, so one reusable line buffer
    // plus to_chars is both deterministic and substantially cheaper at scale.
    std::string output_line;
    output_line.reserve(128);
    char delay_text[std::numeric_limits<std::uint32_t>::digits10 + 3];
    const auto start = std::chrono::steady_clock::now();
    while (std::getline(input, line)) {
        if (trim(line).empty()) {
            continue;
        }
        const auto comma = line.find(',');
        if (comma == std::string::npos || line.find(',', comma + 1) != std::string::npos) {
            throw std::runtime_error("malformed request CSV row " + std::to_string(summary.rows + 2));
        }
        const auto from_text = trim(std::string_view(line).substr(0, comma));
        const auto to_text = trim(std::string_view(line).substr(comma + 1));
        const auto from = parse_endpoint(architecture, from_text);
        const auto to = parse_endpoint(architecture, to_text);
        std::uint32_t delay = 0;
        bool used_exact = false;
        const int manhattan = std::abs(static_cast<int>(to.x) - static_cast<int>(from.x)) +
                              std::abs(static_cast<int>(to.y) - static_cast<int>(from.y));
        if (exact_router != nullptr && manhattan <= exact_manhattan_threshold) {
            SearchStats route_stats;
            const auto exact_delay = route_heuristic_weight_milli == 1000
                                         ? exact_router->shortest_delay(from, to, &route_stats)
                                         : exact_router->shortest_delay_weighted(
                                               from, to, route_heuristic_weight_milli, &route_stats);
            summary.route_popped += route_stats.popped;
            summary.route_expanded_arcs += route_stats.expanded_arcs;
            if (exact_delay) {
                if (calibrate_short_route) {
                    constexpr std::uint64_t kScale = 1'000'000;
                    const std::uint64_t multiplier = manhattan <= 16 ? 999'000 : 997'750;
                    delay = static_cast<std::uint32_t>(
                        (static_cast<std::uint64_t>(*exact_delay) * multiplier + kScale / 2) /
                        kScale);
                } else {
                    delay = *exact_delay;
                }
                used_exact = true;
                ++summary.exact_rows;
            } else {
                ++summary.exact_unreachable_rows;
            }
        }
        if (!used_exact) {
            delay = estimator.estimate(from, to);
            ++summary.model_rows;
        }
        const auto formatted = std::to_chars(
            std::begin(delay_text), std::end(delay_text), delay);
        if (formatted.ec != std::errc()) {
            throw std::runtime_error("failed to format estimated delay");
        }
        output_line.clear();
        output_line.append(from_text);
        output_line.push_back(',');
        output_line.append(to_text);
        output_line.push_back(',');
        output_line.append(delay_text, formatted.ptr);
        output_line.push_back('\n');
        output.write(output_line.data(), static_cast<std::streamsize>(output_line.size()));
        ++summary.rows;
    }
    if (!input.eof()) {
        throw std::runtime_error("error while reading request CSV");
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("error while writing result CSV");
    }
    summary.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return summary;
}

}  // namespace delay
