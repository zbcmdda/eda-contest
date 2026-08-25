#pragma once

#include "architecture.hpp"
#include "short_features.hpp"

#include <cstddef>
#include <string>

namespace delay {

void dump_short_features_csv(const Architecture& architecture, const ShortFeatures& features,
                             const std::string& input_path, const std::string& output_path,
                             std::size_t offset, std::size_t limit);

}  // namespace delay
