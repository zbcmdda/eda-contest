execute_process(
    COMMAND "${ESTIMATE}" -in "${REQUEST}" -out "${OUTPUT}" --arch "${ARCH}"
    RESULT_VARIABLE estimate_result
    ERROR_VARIABLE estimate_error
)
if(NOT estimate_result EQUAL 0)
    message(FATAL_ERROR "estimate failed: ${estimate_error}")
endif()

if(DEFINED SHORT_OUTPUT AND DEFINED SHORT_EXPECTED)
    execute_process(
        COMMAND "${ESTIMATE}" -in "${REQUEST}" -out "${SHORT_OUTPUT}" --arch "${ARCH}"
                    --short-residual
        RESULT_VARIABLE short_result
        ERROR_VARIABLE short_error
    )
    if(NOT short_result EQUAL 0)
        message(FATAL_ERROR "short residual estimate failed: ${short_error}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files "${SHORT_OUTPUT}" "${SHORT_EXPECTED}"
        RESULT_VARIABLE short_compare_result
    )
    if(NOT short_compare_result EQUAL 0)
        message(FATAL_ERROR "short residual output differs from expected fixture")
    endif()
endif()

if(DEFINED ROUTE_REQUEST AND DEFINED ROUTE_OUTPUT AND DEFINED ROUTE_EXPECTED)
    execute_process(
        COMMAND "${ESTIMATE}" -in "${ROUTE_REQUEST}" -out "${ROUTE_OUTPUT}" --arch "${ARCH}"
                    --short-route
        RESULT_VARIABLE route_result
        ERROR_VARIABLE route_error
    )
    if(NOT route_result EQUAL 0)
        message(FATAL_ERROR "short route estimate failed: ${route_error}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files "${ROUTE_OUTPUT}" "${ROUTE_EXPECTED}"
        RESULT_VARIABLE route_compare_result
    )
    if(NOT route_compare_result EQUAL 0)
        message(FATAL_ERROR "short route output differs from expected fixture")
    endif()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUTPUT}" "${EXPECTED}"
    RESULT_VARIABLE compare_result
)
if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR "estimate output differs from expected fixture")
endif()

if(DEFINED OUTPUT128)
    execute_process(
        COMMAND "${ESTIMATE}" -in "${REQUEST}" -out "${OUTPUT128}" --arch "${ARCH}"
                    --model-trees 128
        RESULT_VARIABLE explicit_result
        ERROR_VARIABLE explicit_error
    )
    if(NOT explicit_result EQUAL 0)
        message(FATAL_ERROR "explicit 128-tree estimate failed: ${explicit_error}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUTPUT128}" "${EXPECTED}"
        RESULT_VARIABLE explicit_compare_result
    )
    if(NOT explicit_compare_result EQUAL 0)
        message(FATAL_ERROR "explicit 128-tree output differs from expected fixture")
    endif()
endif()
