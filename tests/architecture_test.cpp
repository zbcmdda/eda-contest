#include "architecture.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("expected architecture directory");
        }
        const auto architecture = delay::Architecture::load(argv[1]);
        expect(architecture.width() == 120, "unexpected grid width");
        expect(architecture.height() == 550, "unexpected grid height");
        expect(architecture.active_site_count() == 60200, "unexpected active site count");
        expect(architecture.ports().size() == 496, "unexpected port count");
        expect(architecture.arc_count() == 8192, "unexpected Arc count");
        expect(architecture.nets().size() == 160, "unexpected Net count");
        expect(architecture.line_gaps().size() == 19, "unexpected Line Gap count");
        expect(architecture.block_gaps().size() == 7, "unexpected Block Gap count");
        expect(architecture.active(16, 0), "expected active SRB missing");
        expect(!architecture.active(15, 0), "Block Gap SRB should be absent");
        expect(!architecture.active(76, 500), "Block Gap SRB should be absent");

        const auto east = architecture.resolve_net(18, 95, architecture.port_id("ZQE[5]"));
        expect(east && east->x == 22 && east->y == 95, "regular Net endpoint mismatch");
        expect(east->to == architecture.port_id("IQE[5]"), "regular Net target port mismatch");
        expect(east->delay == 78, "Line Gap delay mismatch");

        const auto across_block = architecture.resolve_net(75, 25, architecture.port_id("ZSEA[0]"));
        expect(across_block && across_block->x == 90 && across_block->y == 25,
               "crossable Block endpoint mismatch");
        expect(across_block->delay == 292, "crossable Block delay mismatch");

        const auto back_across_block = architecture.resolve_net(90, 25, architecture.port_id("ZSWA[0]"));
        expect(back_across_block && back_across_block->x == 75 && back_across_block->y == 25,
               "reverse crossable Block endpoint mismatch");
        expect(back_across_block->delay == 292, "reverse crossable Block delay mismatch");

        const auto long_across_block = architecture.resolve_net(75, 25, architecture.port_id("ZLE[0]"));
        expect(long_across_block && long_across_block->x == 99 && long_across_block->y == 25,
               "long crossable Block endpoint mismatch");
        expect(long_across_block->delay == 390, "combined Block/Line delay mismatch");

        const auto blocked = architecture.resolve_net(16, 50, architecture.port_id("ZSWA[0]"));
        expect(!blocked, "Net ending in non-crossable Block should disconnect");
        std::cout << "architecture_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "architecture_test: " << error.what() << '\n';
        return 1;
    }
}
