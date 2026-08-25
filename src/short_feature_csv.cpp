#include "short_feature_csv.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace delay {
namespace {
std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' || text.front() == '\r')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) text.remove_suffix(1);
    return text;
}
}

void dump_short_features_csv(const Architecture& architecture, const ShortFeatures& features,
                             const std::string& input_path, const std::string& output_path,
                             std::size_t offset, std::size_t limit) {
    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("cannot open short feature input: " + input_path);
    std::string line;
    if (!std::getline(input, line) || trim(line) != "From,To") throw std::runtime_error("unexpected request CSV header");
    for (std::size_t index = 0; index < offset; ++index) if (!std::getline(input, line))
        throw std::runtime_error("short feature offset exceeds input rows");
    std::ofstream output(output_path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open short feature output: " + output_path);
    output << "From,To";
    for (const auto* name : ShortFeatures::names()) output << ',' << name;
    output << '\n';
    std::size_t rows = 0;
    while (rows < limit && std::getline(input, line)) {
        if (trim(line).empty()) continue;
        const auto comma = line.find(',');
        if (comma == std::string::npos || line.find(',', comma + 1) != std::string::npos)
            throw std::runtime_error("malformed request CSV row");
        const auto from_text = trim(std::string_view(line).substr(0, comma));
        const auto to_text = trim(std::string_view(line).substr(comma + 1));
        const auto values = features.feature_for(parse_endpoint(architecture, from_text), parse_endpoint(architecture, to_text));
        output << from_text << ',' << to_text;
        for (const auto value : values) output << ',' << value;
        output << '\n';
        ++rows;
    }
    if (!output) throw std::runtime_error("failed writing short feature CSV");
}
}  // namespace delay
