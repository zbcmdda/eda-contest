#include "architecture.hpp"
#include "router.hpp"
#include "short_features.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected architecture directory");
        const auto architecture = delay::Architecture::load(argv[1]);
        const delay::ShortFeatures features(architecture);
        const auto from = delay::parse_endpoint(architecture, "SRB_18_95/A_FQ0");
        const auto to = delay::parse_endpoint(architecture, "SRB_38_57/ZDN[7]");
        const auto first = features.feature_for(from, to);
        const auto second = features.feature_for(from, to);
        if (first != second) throw std::runtime_error("short features are not deterministic");
        if (first[0] < 0 || first[1] < 0 || first[17] < 0 || first[23] < 0)
            throw std::runtime_error("short feature counts are invalid");
        if (first[3] >= 0 && first[4] >= 0 && first[3] > first[4])
            throw std::runtime_error("first candidates are not stably sorted");
        std::cout << "short_features_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "short_features_test: " << error.what() << '\n';
        return 1;
    }
}
