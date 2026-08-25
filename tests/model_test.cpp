#include "architecture.hpp"
#include "benchmark.hpp"
#include "ml_estimator.hpp"
#include "router.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("expected architecture directory and answer CSV");
        }
        const auto architecture = delay::Architecture::load(argv[1]);
        const delay::MlEstimator estimator(architecture);
        const delay::MlEstimator explicit_default(architecture, 128);
        const delay::MlEstimator short_residual(architecture, 128, true);
        const auto expect_invalid_tree_limit = [&architecture](std::size_t limit) {
            try {
                const delay::MlEstimator invalid(architecture, limit);
                (void)invalid;
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        expect(estimator.estimate(
                   delay::parse_endpoint(architecture, "SRB_84_356/ZLE[0]"),
                   delay::parse_endpoint(architecture, "SRB_46_116/ZSSB[6]")) ==
                   explicit_default.estimate(
                       delay::parse_endpoint(architecture, "SRB_84_356/ZLE[0]"),
                       delay::parse_endpoint(architecture, "SRB_46_116/ZSSB[6]")),
               "explicit 128-tree model changed default output");
        expect(expect_invalid_tree_limit(0), "zero tree limit was accepted");
        expect(expect_invalid_tree_limit(129), "tree limit above model size was accepted");
        const std::array<std::array<std::string, 3>, 5> cases{{
            {{"SRB_84_356/ZLE[0]", "SRB_46_116/ZSSB[6]", "3597"}},
            {{"SRB_109_8/ZSNB[3]", "SRB_71_483/A_H5", "6100"}},
            {{"SRB_4_269/ZLS[0]", "SRB_18_373/ZSWB[0]", "1704"}},
            {{"SRB_55_537/ZDS[6]", "SRB_39_259/ZSSA[6]", "3493"}},
            {{"SRB_80_492/S_CY", "SRB_101_435/S_A5", "1426"}},
        }};
        for (const auto& test : cases) {
            const auto from = delay::parse_endpoint(architecture, test[0]);
            const auto to = delay::parse_endpoint(architecture, test[1]);
            expect(estimator.estimate(from, to) == static_cast<std::uint32_t>(std::stoul(test[2])),
                   "generated model prediction changed");
        }
        const auto same = delay::parse_endpoint(architecture, cases[0][0]);
        expect(estimator.estimate(same, same) == 0, "same-endpoint delay must be zero");

        // Keep regression tests inside the development range; the final
        // 100,000 rows are reserved for a single final evaluation.
        const auto summary = delay::benchmark_model(
            architecture, estimator, argv[2], 1000, 1, 800000);
        const auto short_summary = delay::benchmark_model(
            architecture, short_residual, argv[2], 1000, 1, 800000);
        expect(summary.rows == 1000, "model benchmark row count mismatch");
        expect(summary.official_score > 90.0, "model regression score is unexpectedly low");
        expect(short_summary.rows == summary.rows, "short residual benchmark row count mismatch");
        expect(short_summary.official_score > summary.official_score,
               "short residual model did not improve the development sample");
        std::cout << "model_test: ok; score=" << summary.official_score
                  << "; short_score=" << short_summary.official_score << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_test: " << error.what() << '\n';
        return 1;
    }
}
