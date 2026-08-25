#include "benchmark.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
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

std::uint32_t parse_delay(std::string_view text) {
    text = trim(text);
    std::uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid delay in answer CSV");
    }
    return value;
}

}  // namespace

BenchmarkSummary benchmark_answers(
    const Architecture& architecture,
    Router& router,
    const std::string& path,
    std::size_t limit,
    std::size_t stride) {
    if (stride == 0) {
        throw std::runtime_error("benchmark stride must be positive");
    }
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open answer CSV: " + path);
    }
    std::string line;
    if (!std::getline(stream, line) || trim(line) != "From,To,delay") {
        throw std::runtime_error("unexpected answer CSV header");
    }

    BenchmarkSummary summary;
    std::size_t source_row = 0;
    const auto start = std::chrono::steady_clock::now();
    while (summary.rows < limit && std::getline(stream, line)) {
        if (source_row++ % stride != 0) {
            continue;
        }
        const auto first = line.find(',');
        const auto second = line.find(',', first == std::string::npos ? first : first + 1);
        if (first == std::string::npos || second == std::string::npos ||
            line.find(',', second + 1) != std::string::npos) {
            throw std::runtime_error("malformed answer CSV row");
        }
        const auto from = parse_endpoint(architecture, line.substr(0, first));
        const auto to = parse_endpoint(architecture, line.substr(first + 1, second - first - 1));
        const auto golden = parse_delay(std::string_view(line).substr(second + 1));
        SearchStats stats;
        const auto estimated = router.shortest_delay(from, to, &stats);
        if (!estimated || *estimated != golden) {
            ++summary.mismatches;
        }
        ++summary.rows;
        summary.total_popped += stats.popped;
        summary.total_expanded_arcs += stats.expanded_arcs;
    }
    summary.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return summary;
}

ModelBenchmarkSummary benchmark_model(
    const Architecture& architecture,
    const MlEstimator& estimator,
    const std::string& path,
    std::size_t limit,
    std::size_t stride,
    std::size_t offset) {
    if (stride == 0) {
        throw std::runtime_error("benchmark stride must be positive");
    }
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open answer CSV: " + path);
    }
    std::string line;
    if (!std::getline(stream, line) || trim(line) != "From,To,delay") {
        throw std::runtime_error("unexpected answer CSV header");
    }
    for (std::size_t index = 0; index < offset; ++index) {
        if (!std::getline(stream, line)) {
            throw std::runtime_error("model benchmark offset exceeds answer row count");
        }
    }

    ModelBenchmarkSummary summary;
    std::size_t source_row = 0;
    double score_sum = 0.0;
    double relative_sum = 0.0;
    std::size_t positive_rows = 0;
    std::size_t within_5 = 0;
    std::size_t within_10 = 0;
    std::size_t within_20 = 0;
    const auto start = std::chrono::steady_clock::now();
    while (summary.rows < limit && std::getline(stream, line)) {
        if (source_row++ % stride != 0) {
            continue;
        }
        const auto first = line.find(',');
        const auto second = line.find(',', first == std::string::npos ? first : first + 1);
        if (first == std::string::npos || second == std::string::npos ||
            line.find(',', second + 1) != std::string::npos) {
            throw std::runtime_error("malformed answer CSV row");
        }
        const auto from = parse_endpoint(architecture, line.substr(0, first));
        const auto to = parse_endpoint(architecture, line.substr(first + 1, second - first - 1));
        const auto golden = parse_delay(std::string_view(line).substr(second + 1));
        const auto predicted = estimator.estimate(from, to);
        if (golden > 0) {
            const double relative = std::abs(static_cast<double>(predicted) - golden) / golden;
            score_sum += 1.0 - std::tanh(4.0 * relative);
            relative_sum += relative;
            within_5 += relative <= 0.05;
            within_10 += relative <= 0.10;
            within_20 += relative <= 0.20;
            ++positive_rows;
        }
        ++summary.rows;
    }
    summary.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (positive_rows != 0) {
        summary.official_score = score_sum / positive_rows * 100.0;
        summary.mean_relative_error = relative_sum / positive_rows;
        summary.within_5_percent = static_cast<double>(within_5) / positive_rows * 100.0;
        summary.within_10_percent = static_cast<double>(within_10) / positive_rows * 100.0;
        summary.within_20_percent = static_cast<double>(within_20) / positive_rows * 100.0;
    }
    return summary;
}

}  // namespace delay
