#include "architecture.hpp"

#include "json.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace delay {
namespace {

std::string json_path(const std::string& directory, const char* filename) {
    return (std::filesystem::path(directory) / filename).string();
}

int checked_int(const json::Value& value, const char* field) {
    const auto integer = value.as_int();
    if (integer < std::numeric_limits<int>::min() || integer > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("integer field out of range: ") + field);
    }
    return static_cast<int>(integer);
}

std::uint16_t checked_u16(const json::Value& value, const char* field) {
    const auto integer = value.as_int();
    if (integer < 0 || integer > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error(std::string("non-negative 16-bit field out of range: ") + field);
    }
    return static_cast<std::uint16_t>(integer);
}

std::int16_t checked_i16(const json::Value& value, const char* field) {
    const auto integer = value.as_int();
    if (integer < std::numeric_limits<std::int16_t>::min() ||
        integer > std::numeric_limits<std::int16_t>::max()) {
        throw std::runtime_error(std::string("16-bit field out of range: ") + field);
    }
    return static_cast<std::int16_t>(integer);
}

bool looks_like_srb_name(const std::string& name, int x, int y) {
    return name == "SRB_" + std::to_string(x) + "_" + std::to_string(y);
}

std::uint64_t port_name_hash(std::string_view name) noexcept {
    // A tiny fixed hash avoids constructing a std::string for every endpoint
    // lookup.  The original map remains the collision-safe fallback below.
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto character : name) {
        hash ^= static_cast<unsigned char>(character);
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

Architecture Architecture::load(const std::string& directory) {
    Architecture result;

    const auto inst_root = json::parse_file(json_path(directory, "SRB_Inst.json"));
    const auto& instances = inst_root.at("Inst").as_array();
    if (instances.empty()) {
        throw std::runtime_error("SRB_Inst.json contains no instances");
    }
    int max_x = -1;
    int max_y = -1;
    for (const auto& value : instances) {
        const int x = checked_int(value.at("x"), "x");
        const int y = checked_int(value.at("y"), "y");
        const auto& name = value.at("name").as_string();
        if (x < 0 || y < 0) {
            throw std::runtime_error("negative SRB coordinate: " + name);
        }
        if (!looks_like_srb_name(name, x, y)) {
            throw std::runtime_error("SRB name/coordinate mismatch: " + name);
        }
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
    }
    result.width_ = max_x + 1;
    result.height_ = max_y + 1;
    result.active_.assign(static_cast<std::size_t>(result.width_) * result.height_, 0);
    for (const auto& value : instances) {
        const int x = checked_int(value.at("x"), "x");
        const int y = checked_int(value.at("y"), "y");
        auto& active = result.active_[static_cast<std::size_t>(y) * result.width_ + x];
        if (active != 0) {
            throw std::runtime_error("duplicate SRB coordinate");
        }
        active = 1;
        ++result.active_site_count_;
    }

    const auto port_root = json::parse_file(json_path(directory, "SRB_Port.json"));
    for (const auto& value : port_root.at("Port").as_array()) {
        const auto& name = value.at("Name").as_string();
        const auto& direction_text = value.at("Direction").as_string();
        Direction direction;
        if (direction_text == "Input") {
            direction = Direction::Input;
        } else if (direction_text == "Output") {
            direction = Direction::Output;
        } else {
            throw std::runtime_error("unknown port direction for " + name + ": " + direction_text);
        }
        if (result.ports_.size() >= std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("too many ports");
        }
        const auto id = static_cast<std::uint16_t>(result.ports_.size());
        if (!result.port_ids_.emplace(name, id).second) {
            throw std::runtime_error("duplicate port: " + name);
        }
        result.port_hash_ids_.emplace(port_name_hash(name), id);
        const bool inter = !name.empty() && (name.front() == 'I' || name.front() == 'Z');
        result.ports_.push_back(Port{name, direction, inter});
    }
    result.arcs_.resize(result.ports_.size());
    result.net_indices_.assign(result.ports_.size(), -1);

    const auto arc_root = json::parse_file(json_path(directory, "SRB_Arc.json"));
    for (const auto& value : arc_root.at("Arcs").as_array()) {
        const auto from = result.port_id(value.at("from").as_string());
        const auto to = result.port_id(value.at("to").as_string());
        result.arcs_[from].push_back(Arc{to, checked_u16(value.at("delay"), "delay")});
        ++result.arc_count_;
    }

    const auto net_root = json::parse_file(json_path(directory, "SRB_Net.json"));
    for (const auto& value : net_root.at("Nets").as_array()) {
        const Net net{
            result.port_id(value.at("from").as_string()),
            result.port_id(value.at("to").as_string()),
            checked_i16(value.at("delta x"), "delta x"),
            checked_i16(value.at("delta y"), "delta y"),
        };
        if (result.net_indices_.at(net.from) != -1) {
            throw std::runtime_error("duplicate Net source port");
        }
        if (result.nets_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int16_t>::max())) {
            throw std::runtime_error("too many Nets");
        }
        result.net_indices_[net.from] = static_cast<std::int16_t>(result.nets_.size());
        result.nets_.push_back(net);
    }

    const auto gap_root = json::parse_file(json_path(directory, "SRB_Gap.json"));
    const auto& gap = gap_root.at("Gap");
    for (const auto& value : gap.at("Line").as_array()) {
        const auto& direction = value.at("direction").as_string();
        if (direction != "horizontal" && direction != "vertical") {
            throw std::runtime_error("unknown line gap direction: " + direction);
        }
        result.line_gaps_.push_back(LineGap{
            direction == "horizontal",
            checked_i16(value.at("site"), "site"),
            checked_u16(value.at("delay"), "delay"),
        });
    }
    for (const auto& value : gap.at("Block").as_array()) {
        result.block_gaps_.push_back(BlockGap{
            checked_i16(value.at("lower"), "lower"),
            checked_i16(value.at("upper"), "upper"),
            checked_i16(value.at("left"), "left"),
            checked_i16(value.at("right"), "right"),
            value.at("vertical crossable").as_bool(),
            checked_u16(value.at("vertical cross delay"), "vertical cross delay"),
            value.at("horizontal crossable").as_bool(),
            checked_u16(value.at("horizontal cross delay"), "horizontal cross delay"),
        });
    }

    result.validate();
    return result;
}

void Architecture::validate() const {
    if (width_ <= 0 || height_ <= 0 ||
        active_.size() != static_cast<std::size_t>(width_) * height_) {
        throw std::runtime_error("invalid SRB grid dimensions");
    }
    if (ports_.empty() || arcs_.size() != ports_.size()) {
        throw std::runtime_error("invalid port/arc table dimensions");
    }

    std::size_t counted_arcs = 0;
    for (std::size_t from = 0; from < arcs_.size(); ++from) {
        for (const auto& arc : arcs_[from]) {
            if (ports_[from].direction != Direction::Input ||
                ports_.at(arc.to).direction != Direction::Output) {
                throw std::runtime_error("Arc is not Input -> Output");
            }
            ++counted_arcs;
        }
    }
    if (counted_arcs != arc_count_) {
        throw std::runtime_error("Arc count mismatch");
    }

    std::unordered_set<std::uint16_t> net_sources;
    for (const auto& net : nets_) {
        if (!ports_.at(net.from).inter || !ports_.at(net.to).inter ||
            ports_.at(net.from).direction != Direction::Output ||
            ports_.at(net.to).direction != Direction::Input) {
            throw std::runtime_error("Net is not inter Output -> inter Input");
        }
        if ((net.dx == 0) == (net.dy == 0)) {
            throw std::runtime_error("Net must move along exactly one axis");
        }
        if (!net_sources.insert(net.from).second) {
            throw std::runtime_error("duplicate Net source port");
        }
    }

    for (const auto& block : block_gaps_) {
        if (block.left > block.right || block.lower > block.upper || block.left < 0 ||
            block.lower < 0 || block.right >= width_ || block.upper >= height_) {
            throw std::runtime_error("invalid Block Gap bounds");
        }
        for (int y = block.lower; y <= block.upper; ++y) {
            for (int x = block.left; x <= block.right; ++x) {
                if (active(x, y)) {
                    throw std::runtime_error("Block Gap contains an active SRB");
                }
            }
        }
    }

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            bool blocked = false;
            for (const auto& block : block_gaps_) {
                blocked = blocked || (x >= block.left && x <= block.right && y >= block.lower && y <= block.upper);
            }
            if (active(x, y) == blocked) {
                throw std::runtime_error("Inst list and Block Gaps do not partition the SRB grid");
            }
        }
    }
}

std::uint16_t Architecture::port_id(std::string_view name) const {
    const auto hash = port_name_hash(name);
    const auto range = port_hash_ids_.equal_range(hash);
    for (auto iterator = range.first; iterator != range.second; ++iterator) {
        const auto port = iterator->second;
        const auto& stored = ports_[port].name;
        if (stored.size() == name.size() &&
            std::equal(stored.begin(), stored.end(), name.begin())) {
            return port;
        }
    }
    // Keep the original lookup as a defensive fallback if an adversarial
    // hash collision or a malformed architecture ever reaches this path.
    const auto iterator = port_ids_.find(std::string(name));
    if (iterator == port_ids_.end()) {
        throw std::runtime_error("unknown port: " + std::string(name));
    }
    return iterator->second;
}

bool Architecture::active(int x, int y) const {
    return x >= 0 && y >= 0 && x < width_ && y < height_ &&
           active_[static_cast<std::size_t>(y) * width_ + x] != 0;
}

std::optional<NetTransition> Architecture::resolve_net(
    int x, int y, std::uint16_t from_port) const {
    if (!active(x, y) || from_port >= net_indices_.size()) {
        return std::nullopt;
    }
    const auto net_index = net_indices_[from_port];
    if (net_index < 0) {
        return std::nullopt;
    }
    const auto& net = nets_[static_cast<std::size_t>(net_index)];
    const bool horizontal = net.dx != 0;
    const int step = horizontal ? (net.dx > 0 ? 1 : -1) : (net.dy > 0 ? 1 : -1);
    int remaining = std::abs(horizontal ? static_cast<int>(net.dx) : static_cast<int>(net.dy));
    int coordinate = horizontal ? x : y;
    std::uint32_t extra_delay = 0;
    if (block_gaps_.size() > 64) {
        throw std::runtime_error("more than 64 Block Gaps are not supported");
    }
    std::uint64_t crossed_blocks = 0;

    // A crossable block consumes physical coordinates without consuming the
    // Net's logical span.  The old implementation advanced one coordinate at
    // a time and re-tested every block on every step.  Find the next eligible
    // block directly, jump over its interval, and leave the final edge case to
    // the same active-site check below.  This is equivalent to the reference
    // loop but removes most work from the production feature path.
    while (remaining > 0) {
        std::size_t next_block = block_gaps_.size();
        int next_distance = remaining + 1;
        for (std::size_t index = 0; index < block_gaps_.size(); ++index) {
            const std::uint64_t bit = std::uint64_t{1} << index;
            if ((crossed_blocks & bit) != 0) {
                continue;
            }
            const auto& block = block_gaps_[index];
            const int band_coordinate = horizontal ? y : x;
            if (band_coordinate < (horizontal ? block.lower : block.left) ||
                band_coordinate > (horizontal ? block.upper : block.right)) {
                continue;
            }
            const bool crossable = horizontal ? block.horizontal_crossable : block.vertical_crossable;
            if (!crossable) {
                continue;
            }
            const int entry = step > 0
                                  ? (horizontal ? block.left : block.lower)
                                  : (horizontal ? block.right : block.upper);
            const int distance = step > 0 ? entry - coordinate : coordinate - entry;
            if (distance > 0 && distance <= remaining && distance < next_distance) {
                next_distance = distance;
                next_block = index;
            }
        }

        if (next_block == block_gaps_.size()) {
            coordinate += step * remaining;
            remaining = 0;
            break;
        }

        const auto& block = block_gaps_[next_block];
        // The entry coordinate itself is part of the skipped block.  There
        // are therefore next_distance - 1 ordinary coordinates before it.
        coordinate += step * (next_distance - 1);
        remaining -= next_distance - 1;
        const int size = horizontal ? block.right - block.left + 1
                                    : block.upper - block.lower + 1;
        coordinate += step * size;
        crossed_blocks |= std::uint64_t{1} << next_block;
        extra_delay += horizontal ? block.horizontal_cross_delay : block.vertical_cross_delay;
    }

    const int target_x = horizontal ? coordinate : x;
    const int target_y = horizontal ? y : coordinate;

    if (!active(target_x, target_y)) {
        return std::nullopt;
    }
    for (const auto& line : line_gaps_) {
        if (horizontal != line.horizontal) {
            const int source = horizontal ? x : y;
            const int target = horizontal ? target_x : target_y;
            if (std::min(source, target) <= line.site && line.site < std::max(source, target)) {
                extra_delay += line.delay;
            }
        }
    }
    if (extra_delay > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("Net extra delay exceeds 16-bit range");
    }
    return NetTransition{
        static_cast<std::int16_t>(target_x),
        static_cast<std::int16_t>(target_y),
        net.to,
        static_cast<std::uint16_t>(extra_delay),
    };
}

std::optional<std::uint16_t> Architecture::arc_delay(
    std::uint16_t from_port, std::uint16_t to_port) const {
    if (from_port >= arcs_.size()) {
        return std::nullopt;
    }
    std::optional<std::uint16_t> best;
    for (const auto& arc : arcs_[from_port]) {
        if (arc.to == to_port && (!best || arc.delay < *best)) {
            best = arc.delay;
        }
    }
    return best;
}

}  // namespace delay
