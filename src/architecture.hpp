#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace delay {

enum class Direction : std::uint8_t { Input, Output };

struct Port {
    std::string name;
    Direction direction;
    bool inter;
};

struct Arc {
    std::uint16_t to;
    std::uint16_t delay;
};

struct Net {
    std::uint16_t from;
    std::uint16_t to;
    std::int16_t dx;
    std::int16_t dy;
};

struct LineGap {
    bool horizontal;
    std::int16_t site;
    std::uint16_t delay;
};

struct BlockGap {
    std::int16_t lower;
    std::int16_t upper;
    std::int16_t left;
    std::int16_t right;
    bool vertical_crossable;
    std::uint16_t vertical_cross_delay;
    bool horizontal_crossable;
    std::uint16_t horizontal_cross_delay;
};

struct NetTransition {
    std::int16_t x;
    std::int16_t y;
    std::uint16_t to;
    std::uint16_t delay;
};

class Architecture {
public:
    static Architecture load(const std::string& directory);

    void validate() const;

    [[nodiscard]] std::uint16_t port_id(std::string_view name) const;
    [[nodiscard]] bool active(int x, int y) const;
    [[nodiscard]] std::optional<NetTransition> resolve_net(
        int x, int y, std::uint16_t from_port) const;
    [[nodiscard]] std::optional<std::uint16_t> arc_delay(
        std::uint16_t from_port, std::uint16_t to_port) const;
    [[nodiscard]] std::size_t active_site_count() const noexcept { return active_site_count_; }
    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] std::size_t arc_count() const noexcept { return arc_count_; }
    [[nodiscard]] const std::vector<Port>& ports() const noexcept { return ports_; }
    [[nodiscard]] const std::vector<std::vector<Arc>>& arcs() const noexcept { return arcs_; }
    [[nodiscard]] const std::vector<Net>& nets() const noexcept { return nets_; }
    [[nodiscard]] const std::vector<LineGap>& line_gaps() const noexcept { return line_gaps_; }
    [[nodiscard]] const std::vector<BlockGap>& block_gaps() const noexcept { return block_gaps_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::size_t active_site_count_ = 0;
    std::size_t arc_count_ = 0;
    std::vector<std::uint8_t> active_;
    std::vector<Port> ports_;
    std::unordered_map<std::string, std::uint16_t> port_ids_;
    std::unordered_multimap<std::uint64_t, std::uint16_t> port_hash_ids_;
    std::vector<std::vector<Arc>> arcs_;
    std::vector<Net> nets_;
    std::vector<std::int16_t> net_indices_;
    std::vector<LineGap> line_gaps_;
    std::vector<BlockGap> block_gaps_;
};

}  // namespace delay
