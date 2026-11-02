# MiniMax H3 W8A8 tensorwise + convrot GEMM and activation quantiser (package gemm).
#   strix_h3_gemm_ref     host reference (ref.hpp), plain C++.
#   strix_h3_gemm_device  device API (api.hpp / kernels.hpp, header-only HIP): carries the HIP runtime, include dirs and
#                         the qualified device compile contract (wave32, no fast math, no fp contraction, correctly
#                         rounded fp32 division/sqrt, IEEE denormals). Consumers also call
#                         strix_h3_gemm_device_target(<target>) to set HIP_STANDARD 20 and HIP_ARCHITECTURES gfx1151.
# Tests: strix-h3-gemm-ref-tests (CPU, always), strix-h3-gemm-gpu-tests and strix-h3-gemm-link-check (HIP).
# Qualification in ctest: -DSTRIX_H3_QUALIFY=ON with STRIX_H3_GEMM_DUMPS_DIR, STRIX_H3_QUANT_SET_DIR (CPU) and, for the GPU
# qualification (also needs STRIX_H3_GPU_CTEST=ON), STRIX_H3_CKPT; missing paths are a configure error (skips never pass).
add_library(strix_h3_gemm_ref INTERFACE)
target_link_libraries(strix_h3_gemm_ref INTERFACE strix_h3_common)
if(MSVC)
    target_compile_options(strix_h3_gemm_ref INTERFACE /fp:strict)
endif()

option(STRIX_H3_QUALIFY "Register the H3 gemm qualification runs (--qualify) in ctest" OFF)
set(STRIX_H3_GEMM_DUMPS_DIR "" CACHE PATH "H3 gemm python cross-check dumps (qualification)")
set(STRIX_H3_QUANT_SET_DIR "" CACHE PATH "H3 quantiser fixture set (qualification)")
set(STRIX_H3_CKPT "" CACHE FILEPATH "H3 int8 convrot checkpoint (GPU qualification)")
if(STRIX_H3_QUALIFY AND (NOT IS_DIRECTORY "${STRIX_H3_GEMM_DUMPS_DIR}" OR NOT IS_DIRECTORY "${STRIX_H3_QUANT_SET_DIR}"))
    message(FATAL_ERROR "STRIX_H3_QUALIFY needs STRIX_H3_GEMM_DUMPS_DIR and STRIX_H3_QUANT_SET_DIR")
endif()

if(BUILD_TESTING)
    add_executable(strix-h3-gemm-ref-tests tests/h3/gemm/ref_tests.cpp)
    target_link_libraries(strix-h3-gemm-ref-tests PRIVATE strix_h3_gemm_ref)
    strix_target_warnings(strix-h3-gemm-ref-tests)
    add_test(NAME strix-h3-gemm-ref-tests COMMAND strix-h3-gemm-ref-tests)
    set_tests_properties(strix-h3-gemm-ref-tests PROPERTIES TIMEOUT 120)
    if(STRIX_H3_QUALIFY)
        add_test(NAME strix-h3-gemm-ref-qualify COMMAND strix-h3-gemm-ref-tests --qualify "${STRIX_H3_GEMM_DUMPS_DIR}" "${STRIX_H3_QUANT_SET_DIR}")
        set_tests_properties(strix-h3-gemm-ref-qualify PROPERTIES LABELS "h3;qualify" TIMEOUT 300)
    endif()
endif()

if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_h3_gemm_device INTERFACE)
    target_link_libraries(strix_h3_gemm_device INTERFACE strix_h3_gemm_ref "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_h3_gemm_device INTERFACE "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_options(strix_h3_gemm_device INTERFACE
        $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee;-fno-gpu-flush-denormals-to-zero>)
    function(strix_h3_gemm_device_target target)
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(${target} PRIVATE strix_h3_gemm_device)
    endfunction()

    add_executable(strix-h3-gemm-link-check tests/h3/gemm/link_a.hip tests/h3/gemm/link_b.hip)
    strix_h3_gemm_device_target(strix-h3-gemm-link-check)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/h3/gemm/gpu_tests.hip")
        add_executable(strix-h3-gemm-gpu-tests tests/h3/gemm/gpu_tests.hip)
        strix_h3_gemm_device_target(strix-h3-gemm-gpu-tests)
        target_compile_options(strix-h3-gemm-gpu-tests PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Rpass-analysis=kernel-resource-usage>)
    endif()
    if(BUILD_TESTING)
        # link check: no GPU work (main only takes addresses)
        add_test(NAME strix-h3-gemm-link-check COMMAND strix-h3-gemm-link-check)
        set_tests_properties(strix-h3-gemm-link-check PROPERTIES LABELS "h3" TIMEOUT 60)
        # GPU tests: labelled `gpu` (skip with `ctest -LE gpu`); disabled unless STRIX_H3_GPU_CTEST=ON, so a plain ctest
        # never touches the GPU. Fixtures through STRIX_H3_GEMM_DUMPS / STRIX_H3_QUANT_SET / H3_CKPT (plain run).
        option(STRIX_H3_GPU_CTEST "Enable the H3 GPU tests in ctest (needs the GPU)" OFF)
        add_test(NAME strix-h3-gemm-gpu-tests COMMAND strix-h3-gemm-gpu-tests)
        set_tests_properties(strix-h3-gemm-gpu-tests PROPERTIES LABELS "gpu;h3" TIMEOUT 300)
        if(STRIX_H3_QUALIFY AND STRIX_H3_GPU_CTEST)
            if(NOT EXISTS "${STRIX_H3_CKPT}")
                message(FATAL_ERROR "the GPU qualification needs STRIX_H3_CKPT")
            endif()
            add_test(NAME strix-h3-gemm-gpu-qualify COMMAND strix-h3-gemm-gpu-tests --qualify "${STRIX_H3_GEMM_DUMPS_DIR}")
            set_tests_properties(strix-h3-gemm-gpu-qualify PROPERTIES LABELS "gpu;h3;qualify" TIMEOUT 300
                ENVIRONMENT "STRIX_H3_QUANT_SET=${STRIX_H3_QUANT_SET_DIR};H3_CKPT=${STRIX_H3_CKPT}")
        endif()
        if(NOT STRIX_H3_GPU_CTEST)
            set_tests_properties(strix-h3-gemm-gpu-tests PROPERTIES DISABLED TRUE)
        endif()
    endif()
endif()
