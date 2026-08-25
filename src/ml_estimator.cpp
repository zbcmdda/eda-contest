#include "ml_estimator.hpp"

#include "generated_medium_short_model.hpp"
#include "generated_model.hpp"
#include "generated_ultrashort_model.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace delay {
namespace {

struct RelaxedMove {
    int delta;
    std::uint16_t cost;
};

std::vector<std::uint32_t> build_relaxed_distances(
    int maximum_delta, const std::vector<RelaxedMove>& moves) {
    int maximum_move = 1;
    for (const auto& move : moves) {
        maximum_move = std::max(maximum_move, std::abs(move.delta));
    }
    const int extent = maximum_delta + maximum_move;
    const auto infinite = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> distances(static_cast<std::size_t>(extent * 2 + 1), infinite);
    using Item = std::pair<std::uint32_t, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    distances[extent] = 0;
    queue.emplace(0, 0);
    while (!queue.empty()) {
        const auto [distance, coordinate] = queue.top();
        queue.pop();
        if (distance != distances[static_cast<std::size_t>(coordinate + extent)]) {
            continue;
        }
        for (const auto& move : moves) {
            const int target = coordinate + move.delta;
            if (target < -extent || target > extent) {
                continue;
            }
            const std::uint32_t candidate = distance + move.cost;
            auto& current = distances[static_cast<std::size_t>(target + extent)];
            if (candidate < current) {
                current = candidate;
                queue.emplace(candidate, target);
            }
        }
    }
    return std::vector<std::uint32_t>(
        distances.begin() + maximum_move,
        distances.begin() + maximum_move + maximum_delta * 2 + 1);
}

int sign_code(int value) {
    return (value > 0 ? 1 : value < 0 ? -1 : 0) + 1;
}

int port_kind_code(std::string_view name) {
    if (name.empty()) {
        return -1;
    }
    switch (name.front()) {
        case 'A': return 0;
        case 'S': return 1;
        case 'Z': return 2;
        default: return -1;
    }
}

int span_kind_code(std::string_view name) {
    if (name.size() < 2 || (name.front() != 'I' && name.front() != 'Z')) {
        return 3;
    }
    switch (name[1]) {
        case 'D': return 0;
        case 'L': return 1;
        case 'S': return 2;
        default: return -1;
    }
}

int span_value(std::string_view name) {
    if (name.size() < 2 || (name.front() != 'I' && name.front() != 'Z')) {
        return 0;
    }
    switch (name[1]) {
        case 'S': return 1;
        case 'D': return 2;
        case 'Q': return 4;
        case 'L': return 10;
        default: return 0;
    }
}

int direction_code(std::string_view name) {
    if (name.size() < 3 || (name.front() != 'I' && name.front() != 'Z')) {
        return 4;
    }
    switch (name[2]) {
        case 'E': return 0;
        case 'N': return 1;
        case 'S': return 2;
        case 'W': return 3;
        default: return 4;
    }
}

int bit_index(std::string_view name) {
    const auto left = name.find('[');
    const auto right = name.find(']', left == std::string_view::npos ? left : left + 1);
    if (left == std::string_view::npos || right == std::string_view::npos) {
        return -1;
    }
    int value = -1;
    const auto result = std::from_chars(name.data() + left + 1, name.data() + right, value);
    return result.ec == std::errc() && result.ptr == name.data() + right ? value : -1;
}

bool inter_port(std::string_view name) {
    return !name.empty() && (name.front() == 'I' || name.front() == 'Z');
}

std::uint32_t gap_difference(const std::vector<std::uint32_t>& prefix, int left, int right) {
    return prefix[std::max(left, right)] - prefix[std::min(left, right)];
}

}  // namespace

MlEstimator::MlEstimator(const Architecture& architecture, std::size_t tree_limit,
                         bool short_residual, int short_residual_threshold)
    : architecture_(architecture), tree_limit_(tree_limit),
      short_residual_threshold_(short_residual_threshold) {
    if (tree_limit_ == 0 || tree_limit_ > 128) {
        throw std::invalid_argument("model tree limit must be between 1 and 128");
    }
    if (short_residual_threshold_ < 0 || short_residual_threshold_ > 128) {
        throw std::invalid_argument("short residual threshold must be between 0 and 128");
    }
    const auto infinite = std::numeric_limits<std::uint16_t>::max();
    final_delays_.assign(architecture.ports().size(), infinite);
    first_routing_delays_.assign(architecture.ports().size(), infinite);
    std::vector<std::int16_t> net_by_port(architecture.ports().size(), -1);
    for (std::size_t index = 0; index < architecture.nets().size(); ++index) {
        net_by_port[architecture.nets()[index].from] = static_cast<std::int16_t>(index);
    }
    std::vector<std::uint16_t> minimum_net_arc(architecture.nets().size(), infinite);
    for (std::size_t from = 0; from < architecture.arcs().size(); ++from) {
        for (const auto& arc : architecture.arcs()[from]) {
            final_delays_[arc.to] = std::min(final_delays_[arc.to], arc.delay);
            const auto net_index = net_by_port[arc.to];
            if (net_index >= 0) {
                first_routing_delays_[from] = std::min(first_routing_delays_[from], arc.delay);
                minimum_net_arc[static_cast<std::size_t>(net_index)] = std::min(
                    minimum_net_arc[static_cast<std::size_t>(net_index)], arc.delay);
            }
        }
    }

    port_metadata_.resize(architecture.ports().size());
    for (std::size_t port = 0; port < architecture.ports().size(); ++port) {
        const auto& name = architecture.ports()[port].name;
        auto& metadata = port_metadata_[port];
        metadata[0] = port_kind_code(name);
        metadata[1] = span_kind_code(name);
        metadata[2] = span_value(name);
        metadata[3] = direction_code(name);
        metadata[4] = bit_index(name);
        metadata[5] = inter_port(name);
        metadata[6] = model::category_code(
            model::kSourcePortCategories, static_cast<std::int32_t>(port));
        metadata[7] = model::category_code(
            model::kTargetPortCategories, static_cast<std::int32_t>(port));
        metadata[8] = model::category_code(
            model::kEffectiveSourcePortCategories, static_cast<std::int32_t>(port));
    }
    const auto port_count = architecture.ports().size();
    endpoint_pair_categories_.resize(port_count * port_count);
    for (std::size_t effective = 0; effective < port_count; ++effective) {
        for (std::size_t target = 0; target < port_count; ++target) {
            const auto raw_pair = static_cast<std::int32_t>(effective * port_count + target);
            endpoint_pair_categories_[effective * port_count + target] = model::category_code(
                model::kEndpointPairCategories, raw_pair);
        }
    }
    if (short_residual) {
        short_features_ = std::make_unique<ShortFeatures>(architecture);
        medium_short_port_categories_.resize(port_count);
        ultrashort_port_categories_.resize(port_count);
        for (std::size_t port = 0; port < port_count; ++port) {
            medium_short_port_categories_[port] = {
                medium_short_model::category_code(
                    medium_short_model::kSourcePortCategories, static_cast<std::int32_t>(port)),
                medium_short_model::category_code(
                    medium_short_model::kTargetPortCategories, static_cast<std::int32_t>(port)),
                medium_short_model::category_code(
                    medium_short_model::kEffectiveSourcePortCategories,
                    static_cast<std::int32_t>(port)),
            };
            ultrashort_port_categories_[port] = {
                ultrashort_model::category_code(
                    ultrashort_model::kSourcePortCategories, static_cast<std::int32_t>(port)),
                ultrashort_model::category_code(
                    ultrashort_model::kTargetPortCategories, static_cast<std::int32_t>(port)),
                ultrashort_model::category_code(
                    ultrashort_model::kEffectiveSourcePortCategories,
                    static_cast<std::int32_t>(port)),
            };
        }
        medium_short_endpoint_pair_categories_.resize(port_count * port_count);
        ultrashort_endpoint_pair_categories_.resize(port_count * port_count);
        for (std::size_t effective = 0; effective < port_count; ++effective) {
            for (std::size_t target = 0; target < port_count; ++target) {
                const auto raw_pair = static_cast<std::int32_t>(effective * port_count + target);
                medium_short_endpoint_pair_categories_[effective * port_count + target] =
                    medium_short_model::category_code(
                        medium_short_model::kEndpointPairCategories, raw_pair);
                ultrashort_endpoint_pair_categories_[effective * port_count + target] =
                    ultrashort_model::category_code(
                        ultrashort_model::kEndpointPairCategories, raw_pair);
            }
        }
    }

    std::vector<RelaxedMove> horizontal_moves;
    std::vector<RelaxedMove> vertical_moves;
    for (std::size_t index = 0; index < architecture.nets().size(); ++index) {
        const auto cost = minimum_net_arc[index];
        if (cost == infinite) {
            continue;
        }
        const auto& net = architecture.nets()[index];
        auto& moves = net.dx != 0 ? horizontal_moves : vertical_moves;
        const int delta = net.dx != 0 ? net.dx : net.dy;
        moves.push_back(RelaxedMove{delta, cost});
        for (const auto& block : architecture.block_gaps()) {
            const bool horizontal = net.dx != 0;
            const bool crossable = horizontal ? block.horizontal_crossable : block.vertical_crossable;
            if (!crossable) {
                continue;
            }
            const int size = horizontal ? block.right - block.left + 1
                                        : block.upper - block.lower + 1;
            const auto delay = horizontal ? block.horizontal_cross_delay : block.vertical_cross_delay;
            moves.push_back(RelaxedMove{
                delta + (delta > 0 ? size : -size),
                static_cast<std::uint16_t>(cost + delay),
            });
        }
    }
    relaxed_x_ = build_relaxed_distances(architecture.width() - 1, horizontal_moves);
    relaxed_y_ = build_relaxed_distances(architecture.height() - 1, vertical_moves);

    vertical_gap_prefix_.resize(architecture.width());
    for (int x = 0; x < architecture.width(); ++x) {
        for (const auto& line : architecture.line_gaps()) {
            if (!line.horizontal && line.site < x) {
                vertical_gap_prefix_[x] += line.delay;
            }
        }
    }
    horizontal_gap_prefix_.resize(architecture.height());
    for (int y = 0; y < architecture.height(); ++y) {
        for (const auto& line : architecture.line_gaps()) {
            if (line.horizontal && line.site < y) {
                horizontal_gap_prefix_[y] += line.delay;
            }
        }
    }
}

std::array<double, 51> MlEstimator::features(
    const Endpoint& from, const Endpoint& to, std::uint16_t* effective_source_output) const {
    std::array<double, 51> result{};
    const int dx = to.x - from.x;
    const int dy = to.y - from.y;
    const int absolute_x = std::abs(dx);
    const int absolute_y = std::abs(dy);
    result[0] = from.x;
    result[1] = from.y;
    result[2] = to.x;
    result[3] = to.y;
    result[4] = dx;
    result[5] = dy;
    result[6] = absolute_x;
    result[7] = absolute_y;
    result[8] = absolute_x + absolute_y;
    result[9] = absolute_x / 10;
    result[10] = absolute_x % 10;
    result[11] = absolute_y / 12;
    result[12] = absolute_y % 12;
    result[13] = sign_code(dx);
    result[14] = sign_code(dy);
    const auto& source_metadata = port_metadata_[from.port];
    const auto& target_metadata = port_metadata_[to.port];
    result[15] = source_metadata[6];
    result[16] = target_metadata[7];
    result[17] = source_metadata[0];
    result[18] = source_metadata[1];
    result[19] = source_metadata[2];
    result[20] = source_metadata[3];
    result[21] = source_metadata[4];
    result[22] = source_metadata[5];
    result[23] = target_metadata[0];
    result[24] = target_metadata[1];
    result[25] = target_metadata[2];
    result[26] = target_metadata[3];
    result[27] = target_metadata[4];
    result[28] = target_metadata[5];

    int after_x = from.x;
    int after_y = from.y;
    std::uint16_t effective_source_port = from.port;
    std::uint32_t source_net_delay = 0;
    if (architecture_.ports()[from.port].direction == Direction::Output &&
        architecture_.ports()[from.port].inter) {
        const auto transition = architecture_.resolve_net(from.x, from.y, from.port);
        if (transition) {
            after_x = transition->x;
            after_y = transition->y;
            effective_source_port = transition->to;
            source_net_delay = transition->delay;
        }
    }
    if (effective_source_output != nullptr) {
        *effective_source_output = effective_source_port;
    }
    const int after_dx = to.x - after_x;
    const int after_dy = to.y - after_y;
    result[29] = after_x;
    result[30] = after_y;
    result[31] = after_dx;
    result[32] = after_dy;
    result[33] = std::abs(after_dx);
    result[34] = std::abs(after_dy);
    result[35] = source_net_delay;
    result[36] = port_metadata_[effective_source_port][8];
    const auto final_delay = final_delays_[to.port] == std::numeric_limits<std::uint16_t>::max()
                                 ? 0
                                 : final_delays_[to.port];
    const auto endpoint_pair = static_cast<std::int32_t>(effective_source_port) *
                                   static_cast<std::int32_t>(architecture_.ports().size()) +
                               to.port;
    result[37] = endpoint_pair_categories_[static_cast<std::size_t>(endpoint_pair)];

    const auto line_x = gap_difference(vertical_gap_prefix_, from.x, to.x);
    const auto line_y = gap_difference(horizontal_gap_prefix_, from.y, to.y);
    result[38] = line_x;
    result[39] = line_y;
    const auto first_delay = first_routing_delays_[from.port] == std::numeric_limits<std::uint16_t>::max()
                                 ? 0
                                 : first_routing_delays_[from.port];
    result[40] = relaxed_x_[static_cast<std::size_t>(dx + architecture_.width() - 1)] +
                 relaxed_y_[static_cast<std::size_t>(dy + architecture_.height() - 1)] +
                 line_x + line_y;
    result[41] = final_delay;
    result[42] = first_delay;
    const auto after_line_x = gap_difference(vertical_gap_prefix_, after_x, to.x);
    const auto after_line_y = gap_difference(horizontal_gap_prefix_, after_y, to.y);
    result[43] = relaxed_x_[static_cast<std::size_t>(after_dx + architecture_.width() - 1)] +
                 relaxed_y_[static_cast<std::size_t>(after_dy + architecture_.height() - 1)] +
                 after_line_x + after_line_y + source_net_delay + final_delay;

    for (std::size_t index = 0; index < architecture_.block_gaps().size(); ++index) {
        const auto& block = architecture_.block_gaps()[index];
        const bool same_band = from.y >= block.lower && from.y <= block.upper &&
                               to.y >= block.lower && to.y <= block.upper;
        const bool crosses = (from.x < block.left && to.x > block.right) ||
                             (from.x > block.right && to.x < block.left);
        result[44 + index] = same_band && crosses;
    }
    return result;
}

std::uint32_t MlEstimator::estimate(const Endpoint& from, const Endpoint& to) const {
    if (from == to) {
        return 0;
    }
    std::uint16_t effective_source_port = from.port;
    const auto base_features = features(from, to, &effective_source_port);
    const double base_prediction = std::expm1(
        model::predict_log_delay(base_features, tree_limit_));
    if (!std::isfinite(base_prediction) || base_prediction < 0.0 ||
        base_prediction > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("ML model produced an invalid delay");
    }
    const auto base_delay = static_cast<std::uint32_t>(std::llround(base_prediction));
    const int manhattan = std::abs(static_cast<int>(to.x) - from.x) +
                          std::abs(static_cast<int>(to.y) - from.y);
    if (!short_features_ || manhattan > short_residual_threshold_) {
        return base_delay;
    }

    const bool ultrashort = manhattan <= 32;
    const auto local = short_features_->feature_for(from, to, ultrashort ? 3 : 2);
    const auto port_count = architecture_.ports().size();
    const auto pair_index = static_cast<std::size_t>(effective_source_port) * port_count + to.port;
    double residual = 0.0;
    if (ultrashort) {
        std::array<double, 82> candidate{};
        std::copy(base_features.begin(), base_features.end(), candidate.begin());
        candidate[15] = ultrashort_port_categories_[from.port][0];
        candidate[16] = ultrashort_port_categories_[to.port][1];
        candidate[36] = ultrashort_port_categories_[effective_source_port][2];
        candidate[37] = ultrashort_endpoint_pair_categories_[pair_index];
        for (std::size_t index = 0; index < local.size(); ++index) {
            candidate[51 + index] = local[index];
        }
        residual = ultrashort_model::predict_residual(candidate);
    } else {
        std::array<double, 75> candidate{};
        std::copy(base_features.begin(), base_features.end(), candidate.begin());
        candidate[15] = medium_short_port_categories_[from.port][0];
        candidate[16] = medium_short_port_categories_[to.port][1];
        candidate[36] = medium_short_port_categories_[effective_source_port][2];
        candidate[37] = medium_short_endpoint_pair_categories_[pair_index];
        for (std::size_t index = 0; index < 24; ++index) {
            candidate[51 + index] = local[index];
        }
        residual = medium_short_model::predict_residual(candidate);
    }
    const double corrected = (static_cast<double>(base_delay) + 1.0) *
                                 std::exp(1.5 * residual) -
                             1.0;
    if (!std::isfinite(corrected) || corrected < 0.0 ||
        corrected > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("short residual model produced an invalid delay");
    }
    return static_cast<std::uint32_t>(std::llround(corrected));
}

}  // namespace delay
