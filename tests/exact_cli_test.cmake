execute_process(
    COMMAND "${ESTIMATE}" --exact-label
            -in "${REQUEST}"
            -out "${OUTPUT}"
            --arch "${ARCH}"
            --limit 1
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "exact CLI failed: ${stdout}${stderr}")
endif()
file(READ "${OUTPUT}" actual)
file(READ "${EXPECTED}" expected)
string(STRIP "${actual}" actual)
string(STRIP "${expected}" expected)
if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "exact CLI output mismatch: ${actual}")
endif()
