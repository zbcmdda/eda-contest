#include "architecture.hpp"
#include "check.hpp"
#include "router.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("expected architecture directory and check CSV");
        }
        const auto architecture = delay::Architecture::load(argv[1]);
        delay::Router router(architecture);
        const auto summary = delay::verify_check_file(architecture, router, argv[2]);
        if (summary.rows != 8) {
            throw std::runtime_error("unexpected golden row count");
        }
        std::cout << "router_test: ok; popped=" << summary.total_popped
                  << "; arcs=" << summary.total_expanded_arcs << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "router_test: " << error.what() << '\n';
        return 1;
    }
}
