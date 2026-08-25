#include "exact_csv.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

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

ExactCsvSummary label_exact_csv(
    const Architecture& architecture,
    Router& router,
    const std::string& input_path,
    const std::string& output_path,
    std::size_t offset,
    std::size_t limit) {
    if (std::filesystem::absolute(input_path).lexically_normal() ==
        std::filesystem::absolute(output_path).lexically_normal()) {
        throw std::runtime_error("exact input and output CSV paths must differ");
    }
    std::ifstream input(input_path);
    if (!input) {
        throw std::runtime_error("cannot open exact request CSV: " + input_path);
    }
    std::string line;
    if (!std::getline(input, line) || trim(line) != "From,To") {
        throw std::runtime_error("unexpected exact request CSV header");
    }
    for (std::size_t index = 0; index < offset; ++index) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("exact label offset exceeds request row count");
        }
    }

    std::ofstream output(output_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open exact answer CSV: " + output_path);
    }
    output << "From,To,delay\n";

    ExactCsvSummary summary;
    const auto start = std::chrono::steady_clock::now();
    while (summary.input_rows < limit && std::getline(input, line)) {
        if (trim(line).empty()) {
            continue;
        }
        const auto comma = line.find(',');
        if (comma == std::string::npos || line.find(',', comma + 1) != std::string::npos) {
            throw std::runtime_error("malformed exact request row");
        }
        const auto from_text = trim(std::string_view(line).substr(0, comma));
        const auto to_text = trim(std::string_view(line).substr(comma + 1));
        const auto from = parse_endpoint(architecture, from_text);
        const auto to = parse_endpoint(architecture, to_text);
        SearchStats stats;
        const auto delay = router.shortest_delay(from, to, &stats);
        ++summary.input_rows;
        summary.total_popped += stats.popped;
        summary.total_expanded_arcs += stats.expanded_arcs;
        if (!delay) {
            ++summary.unreachable_rows;
            continue;
        }
        output << from_text << ',' << to_text << ',' << *delay << '\n';
        ++summary.labeled_rows;
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("error while writing exact answer CSV");
    }
    summary.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return summary;
}

}  // namespace delay
