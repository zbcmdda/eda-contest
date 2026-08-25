cmake_minimum_required(VERSION 3.16)
project(pango_delay_estimator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(delay_core
    src/architecture.cpp
    src/benchmark.cpp
    src/check.cpp
    src/estimate_csv.cpp
    src/exact_csv.cpp
    src/json.cpp
    src/ml_estimator.cpp
    src/router.cpp
    src/short_features.cpp
    src/short_feature_csv.cpp
)
target_include_directories(delay_core PUBLIC src)
target_compile_options(delay_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(estimate src/main.cpp)
target_link_libraries(estimate PRIVATE delay_core)
target_compile_options(estimate PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS estimate RUNTIME DESTINATION .)
install(DIRECTORY arch DESTINATION .)

include(CTest)
if(BUILD_TESTING)
    add_executable(architecture_test tests/architecture_test.cpp)
    target_link_libraries(architecture_test PRIVATE delay_core)
    target_compile_options(architecture_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(
        NAME architecture_test
        COMMAND architecture_test ${CMAKE_CURRENT_SOURCE_DIR}/arch
    )

    add_executable(router_test tests/router_test.cpp)
    target_link_libraries(router_test PRIVATE delay_core)
    target_compile_options(router_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(
        NAME router_test
        COMMAND router_test
                ${CMAKE_CURRENT_SOURCE_DIR}/arch
                ${CMAKE_CURRENT_SOURCE_DIR}/data/delay_estimate_check.csv
    )

    add_executable(model_test tests/model_test.cpp)
    target_link_libraries(model_test PRIVATE delay_core)
    target_compile_options(model_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(
        NAME model_test
        COMMAND model_test
                ${CMAKE_CURRENT_SOURCE_DIR}/arch
                ${CMAKE_CURRENT_SOURCE_DIR}/data/delay_estimate_ans.csv
    )

    add_executable(short_features_test tests/short_features_test.cpp)
    target_link_libraries(short_features_test PRIVATE delay_core)
    target_compile_options(short_features_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(
        NAME short_features_test
        COMMAND short_features_test ${CMAKE_CURRENT_SOURCE_DIR}/arch
    )

    add_test(
        NAME cli_test
        COMMAND ${CMAKE_COMMAND}
                -DESTIMATE=$<TARGET_FILE:estimate>
                -DARCH=${CMAKE_CURRENT_SOURCE_DIR}/arch
                -DREQUEST=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/request.csv
                -DEXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/result.csv
                -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli_test_result.csv
                -DOUTPUT128=${CMAKE_CURRENT_BINARY_DIR}/cli_test_result_128.csv
                -DSHORT_EXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/short_residual_result.csv
                -DSHORT_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli_test_short_residual.csv
                -DROUTE_REQUEST=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/short_route_request.csv
                -DROUTE_EXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/short_route_result.csv
                -DROUTE_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli_test_short_route.csv
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.cmake
    )

    add_test(
        NAME exact_cli_test
        COMMAND ${CMAKE_COMMAND}
                -DESTIMATE=$<TARGET_FILE:estimate>
                -DARCH=${CMAKE_CURRENT_SOURCE_DIR}/arch
                -DREQUEST=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/exact_request.csv
                -DEXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/exact_expected.csv
                -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/exact_cli_test_result.csv
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/exact_cli_test.cmake
    )
endif()
