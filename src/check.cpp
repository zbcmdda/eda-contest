#include "check.hpp"

#include <charconv>
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

std::vector<std::string_view> split(std::string_view text, std::string_view separator) {
    std::vector<std::string_view> parts;
    while (true) {
        const auto position = text.find(separator);
        if (position == std::string_view::npos) {
            parts.push_back(trim(text));
            return parts;
        }
        parts.push_back(trim(text.substr(0, position)));
        text.remove_prefix(position + separator.size());
    }
}

std::uint32_t parse_delay(std::string_view text) {
    text = trim(text);
    std::uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid Min Delay in check CSV");
    }
    return value;
}

std::uint32_t verify_path(
    const Architecture& architecture, const std::vector<Endpoint>& endpoints) {
    std::uint32_t total = 0;
    for (std::size_t index = 1; index < endpoints.size(); ++index) {
        const auto& from = endpoints[index - 1];
        const auto& to = endpoints[index];
        if (from.x == to.x && from.y == to.y) {
            const auto delay = architecture.arc_delay(from.port, to.port);
            if (!delay) {
                throw std::runtime_error("golden Path contains a missing Arc");
            }
            total += *delay;
        } else {
            const auto transition = architecture.resolve_net(from.x, from.y, from.port);
            if (!transition || transition->x != to.x || transition->y != to.y || transition->to != to.port) {
                throw std::runtime_error("golden Path contains a missing or incorrect Net");
            }
            total += transition->delay;
        }
    }
    return total;
}

}  // namespace

CheckSummary verify_check_file(
    const Architecture& architecture, Router& router, const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open check CSV: " + path);
    }
    std::string line;
    if (!std::getline(stream, line) || trim(line) != "From,To,Min Delay,Path") {
        throw std::runtime_error("unexpected check CSV header");
    }

    CheckSummary summary;
    while (std::getline(stream, line)) {
        if (trim(line).empty()) {
            continue;
        }
        const auto first = line.find(',');
        const auto second = line.find(',', first == std::string::npos ? first : first + 1);
        const auto third = line.find(',', second == std::string::npos ? second : second + 1);
        if (first == std::string::npos || second == std::string::npos || third == std::string::npos) {
            throw std::runtime_error("malformed check CSV row");
        }
        const auto from = parse_endpoint(architecture, line.substr(0, first));
        const auto to = parse_endpoint(architecture, line.substr(first + 1, second - first - 1));
        const auto golden = parse_delay(std::string_view(line).substr(second + 1, third - second - 1));
        const auto path_parts = split(std::string_view(line).substr(third + 1), "|");
        std::vector<Endpoint> endpoints;
        endpoints.reserve(path_parts.size());
        for (const auto part : path_parts) {
            endpoints.push_back(parse_endpoint(architecture, std::string(part)));
        }
        if (endpoints.empty() || !(endpoints.front() == from) || !(endpoints.back() == to)) {
            throw std::runtime_error("golden Path endpoints do not match From/To");
        }
        if (verify_path(architecture, endpoints) != golden) {
            throw std::runtime_error("golden Path edge delays do not sum to Min Delay");
        }

        SearchStats stats;
        const auto estimated = router.shortest_delay(from, to, &stats);
        if (!estimated || *estimated != golden) {
            throw std::runtime_error("exact router disagrees with golden Min Delay at row " +
                                     std::to_string(summary.rows + 2));
        }
        ++summary.rows;
        summary.total_popped += stats.popped;
        summary.total_expanded_arcs += stats.expanded_arcs;
    }
    return summary;
}

}  // namespace delay
