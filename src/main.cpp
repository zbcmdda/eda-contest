#include "architecture.hpp"
#include "benchmark.hpp"
#include "check.hpp"
#include "estimate_csv.hpp"
#include "exact_csv.hpp"
#include "ml_estimator.hpp"
#include "router.hpp"
#include "short_feature_csv.hpp"
#include "short_features.hpp"

#include <exception>
#include <filesystem>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string arch_directory;
    bool validate = false;
    bool exact_label = false;
    bool dump_short_features = false;
    bool short_residual = false;
    bool short_route = false;
    int short_residual_threshold = 128;
    std::string check_file;
    std::string benchmark_file;
    std::string model_benchmark_file;
    std::string input_file;
    std::string output_file;
    std::size_t benchmark_limit = 100;
    std::size_t benchmark_stride = 1;
    std::size_t benchmark_offset = 0;
    std::size_t model_trees = 128;
    int exact_manhattan_threshold = -1;
    int fast_route_manhattan_threshold = -1;
    std::uint32_t fast_route_weight_milli = 1500;
};

std::string resolve_arch_directory(const Options& options, const char* executable) {
    if (!options.arch_directory.empty()) {
        return options.arch_directory;
    }
    const std::filesystem::path current = "arch";
    if (std::filesystem::is_regular_file(current / "SRB_Inst.json")) {
        return current.string();
    }
    std::error_code error;
    const auto executable_path = std::filesystem::weakly_canonical(executable, error);
    if (!error) {
        const auto adjacent = executable_path.parent_path() / "arch";
        if (std::filesystem::is_regular_file(adjacent / "SRB_Inst.json")) {
            return adjacent.string();
        }
    }
    throw std::runtime_error(
        "cannot locate architecture files; use --arch DIR or place arch/ beside estimate");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--arch") {
            if (++i == argc) {
                throw std::runtime_error("--arch requires a directory");
            }
            options.arch_directory = argv[i];
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--exact-label") {
            options.exact_label = true;
        } else if (argument == "--dump-short-features") {
            options.dump_short_features = true;
        } else if (argument == "--short-residual") {
            options.short_residual = true;
        } else if (argument == "--short-route") {
            options.short_route = true;
        } else if (argument == "--short-residual-threshold") {
            if (++i == argc) {
                throw std::runtime_error("--short-residual-threshold requires an integer from 0 to 128");
            }
            const std::string value = argv[i];
            std::size_t position = 0;
            long long parsed = 0;
            try {
                parsed = std::stoll(value, &position);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid value for --short-residual-threshold: " + value);
            }
            if (position != value.size() || parsed < 0 || parsed > 128) {
                throw std::runtime_error("invalid value for --short-residual-threshold: " + value);
            }
            options.short_residual = true;
            options.short_residual_threshold = static_cast<int>(parsed);
        } else if (argument == "--check") {
            if (++i == argc) {
                throw std::runtime_error("--check requires a CSV file");
            }
            options.check_file = argv[i];
        } else if (argument == "--benchmark") {
            if (++i == argc) {
                throw std::runtime_error("--benchmark requires an answer CSV file");
            }
            options.benchmark_file = argv[i];
        } else if (argument == "--model-benchmark") {
            if (++i == argc) {
                throw std::runtime_error("--model-benchmark requires an answer CSV file");
            }
            options.model_benchmark_file = argv[i];
        } else if (argument == "--exact-threshold") {
            if (++i == argc) {
                throw std::runtime_error("--exact-threshold requires a non-negative integer");
            }
            const std::string value = argv[i];
            std::size_t position = 0;
            long long parsed = 0;
            try {
                parsed = std::stoll(value, &position);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid value for --exact-threshold: " + value);
            }
            if (position != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
                throw std::runtime_error("invalid value for --exact-threshold: " + value);
            }
            options.exact_manhattan_threshold = static_cast<int>(parsed);
        } else if (argument == "--fast-route-threshold") {
            if (++i == argc) {
                throw std::runtime_error("--fast-route-threshold requires a non-negative integer");
            }
            const std::string value = argv[i];
            std::size_t position = 0;
            long long parsed = 0;
            try {
                parsed = std::stoll(value, &position);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid value for --fast-route-threshold: " + value);
            }
            if (position != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
                throw std::runtime_error("invalid value for --fast-route-threshold: " + value);
            }
            options.fast_route_manhattan_threshold = static_cast<int>(parsed);
        } else if (argument == "--fast-route-weight-milli") {
            if (++i == argc) {
                throw std::runtime_error("--fast-route-weight-milli requires an integer from 1000 to 10000");
            }
            const std::string value = argv[i];
            std::size_t position = 0;
            unsigned long long parsed = 0;
            try {
                parsed = std::stoull(value, &position);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid value for --fast-route-weight-milli: " + value);
            }
            if (position != value.size() || parsed < 1000 || parsed > 10000) {
                throw std::runtime_error("invalid value for --fast-route-weight-milli: " + value);
            }
            options.fast_route_weight_milli = static_cast<std::uint32_t>(parsed);
        } else if (argument == "-in" || argument == "-out") {
            if (++i == argc) {
                throw std::runtime_error(argument + " requires a CSV file");
            }
            if (argument == "-in") {
                options.input_file = argv[i];
            } else {
                options.output_file = argv[i];
            }
        } else if (argument == "--limit" || argument == "--stride" || argument == "--offset" ||
                   argument == "--model-trees") {
            if (++i == argc) {
                throw std::runtime_error(argument + " requires a positive integer");
            }
            const std::string value = argv[i];
            std::size_t parsed = 0;
            std::size_t position = 0;
            try {
                parsed = std::stoull(value, &position);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid value for " + argument + ": " + value);
            }
            if (position != value.size() || (parsed == 0 && argument != "--offset")) {
                throw std::runtime_error("invalid value for " + argument + ": " + value);
            }
            if (argument == "--limit") {
                options.benchmark_limit = parsed;
            } else if (argument == "--stride") {
                options.benchmark_stride = parsed;
            } else if (argument == "--offset") {
                options.benchmark_offset = parsed;
            } else {
                if (parsed == 0 || parsed > 128) {
                    throw std::runtime_error("model tree limit must be between 1 and 128");
                }
                options.model_trees = parsed;
            }
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: estimate -in REQUEST.csv -out RESULT.csv [--arch DIR]\n"
                         "       estimate -in REQUEST.csv -out RESULT.csv [--exact-threshold N]\n"
                         "       estimate -in REQUEST.csv -out RESULT.csv [--fast-route-threshold N]\n"
                         "                [--fast-route-weight-milli 1000..10000]\n"
                         "       estimate -in REQUEST.csv -out RESULT.csv [--short-residual]\n"
                         "       estimate -in REQUEST.csv -out RESULT.csv [--short-route]\n"
                         "       estimate -in REQUEST.csv -out RESULT.csv [--short-residual-threshold N]\n"
                         "       estimate --exact-label -in REQUEST.csv -out ANSWERS.csv\n"
                         "       estimate --dump-short-features -in REQUEST.csv -out FEATURES.csv\n"
                         "       estimate [--validate] [--check FILE] [--benchmark FILE]\n"
                         "                [--model-benchmark FILE] [--limit N] [--stride N]\n"
                         "                [--offset N] [--model-trees N] [--arch DIR]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.input_file.empty() != options.output_file.empty()) {
        throw std::runtime_error("-in and -out must be provided together");
    }
    if ((options.exact_label || options.dump_short_features) && options.input_file.empty()) {
        throw std::runtime_error("offline CSV operations require -in and -out");
    }
    const int route_modes = (options.exact_manhattan_threshold >= 0) +
                            (options.fast_route_manhattan_threshold >= 0) +
                            options.short_route;
    if (route_modes > 1) {
        throw std::runtime_error("exact, fast, and short route modes are mutually exclusive");
    }
    if (!options.validate && options.check_file.empty() && options.benchmark_file.empty() &&
        options.model_benchmark_file.empty() && options.input_file.empty()) {
        throw std::runtime_error("no operation selected; use -in/-out or a validation option");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto architecture = delay::Architecture::load(resolve_arch_directory(options, argv[0]));
        if (options.validate || !options.check_file.empty() || !options.benchmark_file.empty() ||
            !options.model_benchmark_file.empty()) {
            std::cout << "architecture valid\n"
                      << "grid=" << architecture.width() << 'x' << architecture.height() << '\n'
                      << "active_sites=" << architecture.active_site_count() << '\n'
                      << "ports=" << architecture.ports().size() << '\n'
                      << "arcs=" << architecture.arc_count() << '\n'
                      << "nets=" << architecture.nets().size() << '\n'
                      << "line_gaps=" << architecture.line_gaps().size() << '\n'
                      << "block_gaps=" << architecture.block_gaps().size() << '\n';
        }
        if (!options.check_file.empty()) {
            delay::Router router(architecture);
            const auto summary = delay::verify_check_file(architecture, router, options.check_file);
            std::cout << "check_rows=" << summary.rows << '\n'
                      << "search_popped=" << summary.total_popped << '\n'
                      << "expanded_arcs=" << summary.total_expanded_arcs << '\n';
        }
        if (!options.benchmark_file.empty()) {
            delay::Router router(architecture);
            const auto summary = delay::benchmark_answers(
                architecture,
                router,
                options.benchmark_file,
                options.benchmark_limit,
                options.benchmark_stride);
            const double queries_per_second = summary.elapsed_seconds == 0.0
                                                  ? 0.0
                                                  : summary.rows / summary.elapsed_seconds;
            std::cout << "benchmark_rows=" << summary.rows << '\n'
                      << "mismatches=" << summary.mismatches << '\n'
                      << "search_popped=" << summary.total_popped << '\n'
                      << "expanded_arcs=" << summary.total_expanded_arcs << '\n'
                      << "elapsed_seconds=" << summary.elapsed_seconds << '\n'
                      << "queries_per_second=" << queries_per_second << '\n';
        }
        if (!options.model_benchmark_file.empty()) {
            const delay::MlEstimator estimator(
                architecture, options.model_trees, options.short_residual,
                options.short_residual_threshold);
            const auto summary = delay::benchmark_model(
                architecture,
                estimator,
                options.model_benchmark_file,
                options.benchmark_limit,
                options.benchmark_stride,
                options.benchmark_offset);
            const double queries_per_second = summary.elapsed_seconds == 0.0
                                                  ? 0.0
                                                  : summary.rows / summary.elapsed_seconds;
            std::cout << "model_benchmark_rows=" << summary.rows << '\n'
                      << "official_score=" << summary.official_score << '\n'
                      << "mean_relative_error=" << summary.mean_relative_error << '\n'
                      << "within_5_percent=" << summary.within_5_percent << '\n'
                      << "within_10_percent=" << summary.within_10_percent << '\n'
                      << "within_20_percent=" << summary.within_20_percent << '\n'
                      << "elapsed_seconds=" << summary.elapsed_seconds << '\n'
                      << "queries_per_second=" << queries_per_second << '\n';
        }
        if (!options.input_file.empty()) {
            if (options.dump_short_features) {
                const delay::ShortFeatures features(architecture);
                delay::dump_short_features_csv(architecture, features, options.input_file, options.output_file,
                                               options.benchmark_offset, options.benchmark_limit);
            } else if (options.exact_label) {
                delay::Router router(architecture);
                const auto summary = delay::label_exact_csv(
                    architecture,
                    router,
                    options.input_file,
                    options.output_file,
                    options.benchmark_offset,
                    options.benchmark_limit);
                std::cout << "exact_input_rows=" << summary.input_rows << '\n'
                          << "exact_labeled_rows=" << summary.labeled_rows << '\n'
                          << "exact_unreachable_rows=" << summary.unreachable_rows << '\n'
                          << "search_popped=" << summary.total_popped << '\n'
                          << "expanded_arcs=" << summary.total_expanded_arcs << '\n'
                          << "elapsed_seconds=" << summary.elapsed_seconds << '\n';
            } else {
                const delay::MlEstimator estimator(
                    architecture, options.model_trees, options.short_residual,
                    options.short_residual_threshold);
                std::unique_ptr<delay::Router> exact_router;
                const int route_threshold = options.exact_manhattan_threshold >= 0
                                                ? options.exact_manhattan_threshold
                                            : options.fast_route_manhattan_threshold >= 0
                                                ? options.fast_route_manhattan_threshold
                                                : options.short_route ? 32 : -1;
                const auto route_weight = options.exact_manhattan_threshold >= 0
                                              ? std::uint32_t{1000}
                                          : options.short_route
                                              ? std::uint32_t{1200}
                                              : options.fast_route_weight_milli;
                if (route_threshold >= 0) {
                    exact_router = std::make_unique<delay::Router>(architecture);
                }
                const auto summary = delay::estimate_csv(
                    architecture,
                    estimator,
                    options.input_file,
                    options.output_file,
                    exact_router.get(),
                    route_threshold,
                    route_weight,
                    options.short_route);
                if (route_threshold >= 0) {
                    std::cout << "route_threshold=" << route_threshold << '\n'
                              << "route_weight_milli=" << route_weight << '\n'
                              << "exact_rows=" << summary.exact_rows << '\n'
                              << "model_rows=" << summary.model_rows << '\n'
                              << "exact_unreachable_rows=" << summary.exact_unreachable_rows << '\n'
                              << "route_popped=" << summary.route_popped << '\n'
                              << "route_expanded_arcs=" << summary.route_expanded_arcs << '\n'
                              << "elapsed_seconds=" << summary.elapsed_seconds << '\n';
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "estimate: " << error.what() << '\n';
        return 1;
    }
}
