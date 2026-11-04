# Strict g4-v1 target operators; the host layout gate never requires a device runtime.
add_library(strix_gemma4_b_contracts INTERFACE)
target_include_directories(strix_gemma4_b_contracts INTERFACE src include)
target_compile_features(strix_gemma4_b_contracts INTERFACE cxx_std_20)
if(MSVC)
    target_compile_definitions(strix_gemma4_b_contracts INTERFACE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(strix_gemma4_b_contracts INTERFACE /fp:strict /permissive- /Zc:__cplusplus /utf-8)
else()
    target_compile_options(strix_gemma4_b_contracts INTERFACE -fno-fast-math -ffp-contract=off)
endif()
if(NOT STRIX_ENABLE_HIP_RUNTIME AND TARGET strix_gemma4_ref)
    # Compile public device headers, CLI and test host code with the native host compiler.
    # This object-only target deliberately neither links nor invokes the device runtime.
    add_library(strix_gemma4_device_host_compile OBJECT EXCLUDE_FROM_ALL
        src/gemma4/kernels/bench_main.cpp tests/gemma4_gpu/rows.cpp tests/gemma4_a8/device.cpp
        src/gemma4/kernels/a8_teacher_main.cpp)
    target_link_libraries(strix_gemma4_device_host_compile PRIVATE strix_gemma4_b_contracts strix_gemma4_ref)
    strix_target_warnings(strix_gemma4_device_host_compile)
endif()
if(BUILD_TESTING)
    add_executable(strix-gemma4-kv-layout-tests tests/gemma4_gpu/layout.cpp)
    target_link_libraries(strix-gemma4-kv-layout-tests PRIVATE strix_gemma4_b_contracts)
    strix_target_warnings(strix-gemma4-kv-layout-tests)
    add_test(NAME strix-gemma4-kv-layout-tests COMMAND strix-gemma4-kv-layout-tests)
    set_tests_properties(strix-gemma4-kv-layout-tests PROPERTIES TIMEOUT 60)
    if(TARGET strix_gemma4_ref)
        # g4-a8-v1 host contract (quantization, packed group decode, declared order, row invariance).
        add_executable(strix-gemma4-a8-host-tests tests/gemma4_a8/host.cpp)
        target_link_libraries(strix-gemma4-a8-host-tests PRIVATE strix_gemma4_b_contracts strix_gemma4_ref)
        strix_target_warnings(strix-gemma4-a8-host-tests)
        add_test(NAME strix-gemma4-a8-host-tests COMMAND strix-gemma4-a8-host-tests)
        set_tests_properties(strix-gemma4-a8-host-tests PROPERTIES TIMEOUT 60)
    endif()
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_gemma4_device_primitives STATIC src/gemma4/kernels/runtime.hip src/gemma4/kernels/linear.hip
        src/gemma4/kernels/rows_linear.hip src/gemma4/kernels/ple.hip src/gemma4/kernels/attention.hip
        src/gemma4/kernels/a8.hip src/gemma4/kernels/attention_fast_decode.hip
        src/gemma4/kv/transactions.hip src/gemma4/kv/fp16_mirror.hip)
    target_link_libraries(strix_gemma4_device_primitives PUBLIC strix_gemma4_b_contracts "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_gemma4_device_primitives PUBLIC "${STRIX_HIP_INCLUDE_DIR}")
    set(gemma4_b_hip_targets strix_gemma4_device_primitives)
    if(TARGET strix_gemma4_ref)
        add_library(strix_gemma4_device STATIC src/gemma4/kernels/target.hip)
        target_link_libraries(strix_gemma4_device PUBLIC strix_gemma4_device_primitives strix_gemma4_ref)
        list(APPEND gemma4_b_hip_targets strix_gemma4_device)
        add_executable(strix-gemma4-bench src/gemma4/kernels/bench_main.cpp)
        target_link_libraries(strix-gemma4-bench PRIVATE strix_gemma4_device)
        strix_target_warnings(strix-gemma4-bench)
        # g4-a8-v1 teacher-forced fidelity receipts (explicit full-model CLI, not a ctest).
        add_executable(strix-gemma4-a8-teacher src/gemma4/kernels/a8_teacher_main.cpp)
        target_link_libraries(strix-gemma4-a8-teacher PRIVATE strix_gemma4_device)
        strix_target_warnings(strix-gemma4-a8-teacher)
        if(BUILD_TESTING)
            add_executable(strix-gemma4-device-ops-tests tests/gemma4_gpu/operators.hip)
            target_link_libraries(strix-gemma4-device-ops-tests PRIVATE strix_gemma4_device)
            list(APPEND gemma4_b_hip_targets strix-gemma4-device-ops-tests)
            # g4-dattn-f1 (FAST decode attention, global layers) and the per-stream FP16 KV mirror.
            add_executable(strix-gemma4-device-fast-decode-attention-tests tests/gemma4_gpu/fast_decode_attention.hip)
            target_link_libraries(strix-gemma4-device-fast-decode-attention-tests PRIVATE strix_gemma4_device)
            list(APPEND gemma4_b_hip_targets strix-gemma4-device-fast-decode-attention-tests)
            add_test(NAME strix-gemma4-device-fast-decode-attention-tests
                COMMAND strix-gemma4-device-fast-decode-attention-tests)
            set_tests_properties(strix-gemma4-device-fast-decode-attention-tests PROPERTIES TIMEOUT 120
                LABELS "gpu;gemma4")
            add_executable(strix-gemma4-a8-device-tests tests/gemma4_a8/device.cpp)
            target_link_libraries(strix-gemma4-a8-device-tests PRIVATE strix_gemma4_device)
            strix_target_warnings(strix-gemma4-a8-device-tests)
            add_test(NAME strix-gemma4-device-a8-tests COMMAND strix-gemma4-a8-device-tests)
            set_tests_properties(strix-gemma4-device-a8-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4")
            # Fallback schedules (separate quantization launches, plain launches) give the same bytes.
            foreach(variant IN ITEMS "graphs-off|STRIX_G4_GRAPHS=0" "stage-off|STRIX_G4_NORM_STAGE=0"
                    "layer-norm-off|STRIX_G4_LAYER_NORM_FUSION=0" "fusion-off|STRIX_G4_FUSION=0"
                    "store-off|STRIX_G4_STORE_FUSION=0" "multi-off|STRIX_G4_MULTI_LINEAR=0" "half-on|STRIX_G4_A8_HALF=1"
                    "pipe-on|STRIX_G4_A8_PIPE=1" "stream-on|STRIX_G4_A8_STREAM=1"
                    "split-off|STRIX_G4_A8_SPLIT_TREE=0")
                string(REPLACE "|" ";" parts "${variant}")
                list(GET parts 0 suffix)
                list(GET parts 1 setting)
                add_test(NAME strix-gemma4-device-a8-${suffix}-tests COMMAND strix-gemma4-a8-device-tests)
                set_tests_properties(strix-gemma4-device-a8-${suffix}-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                    ENVIRONMENT "${setting}")
            endforeach()
            # Exact many-row linears: byte gate against the eight-row passes (bench: diagnostic timings).
            add_executable(strix-gemma4-device-rows-linear-tests tests/gemma4_gpu/rows_linear.hip)
            target_link_libraries(strix-gemma4-device-rows-linear-tests PRIVATE strix_gemma4_device)
            list(APPEND gemma4_b_hip_targets strix-gemma4-device-rows-linear-tests)
            add_test(NAME strix-gemma4-device-rows-linear-tests COMMAND strix-gemma4-device-rows-linear-tests)
            set_tests_properties(strix-gemma4-device-rows-linear-tests PROPERTIES TIMEOUT 120 LABELS "gpu;gemma4")
            # Prompt-prefill fused epilogues: byte gate against the separate launches (odd and chunk row counts).
            add_executable(strix-gemma4-device-fused-epilogue-tests tests/gemma4_gpu/fused_epilogue.hip)
            target_link_libraries(strix-gemma4-device-fused-epilogue-tests PRIVATE strix_gemma4_device)
            list(APPEND gemma4_b_hip_targets strix-gemma4-device-fused-epilogue-tests)
            add_test(NAME strix-gemma4-device-fused-epilogue-tests COMMAND strix-gemma4-device-fused-epilogue-tests)
            set_tests_properties(strix-gemma4-device-fused-epilogue-tests PROPERTIES TIMEOUT 120 LABELS "gpu;gemma4")
            # Long-span decode attention: shared key/value chunks against the chunk-fastest launches.
            add_executable(strix-gemma4-device-attention-shared-tests tests/gemma4_gpu/attention_shared.hip)
            target_link_libraries(strix-gemma4-device-attention-shared-tests PRIVATE strix_gemma4_device)
            list(APPEND gemma4_b_hip_targets strix-gemma4-device-attention-shared-tests)
            add_test(NAME strix-gemma4-device-attention-shared-tests COMMAND strix-gemma4-device-attention-shared-tests)
            set_tests_properties(strix-gemma4-device-attention-shared-tests PROPERTIES TIMEOUT 120 LABELS "gpu;gemma4")
            add_executable(strix-gemma4-device-rows-tests tests/gemma4_gpu/rows.cpp)
            target_link_libraries(strix-gemma4-device-rows-tests PRIVATE strix_gemma4_device)
            strix_target_warnings(strix-gemma4-device-rows-tests)
            foreach(part ops rows)
                add_test(NAME strix-gemma4-device-${part}-tests COMMAND strix-gemma4-device-${part}-tests)
                set_tests_properties(strix-gemma4-device-${part}-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4")
            endforeach()
            add_test(NAME strix-gemma4-device-rows-off-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_SHARED_WEIGHTS=0;STRIX_G4_FUSION=0;STRIX_G4_GRAPHS=0;STRIX_G4_NORM_GROUP=0;STRIX_G4_CHUNK_DECODE=0;STRIX_G4_KV_POOL=0")
            add_test(NAME strix-gemma4-device-rows-graphs-off-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-graphs-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_GRAPHS=0")
            add_test(NAME strix-gemma4-device-ops-search-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-search-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_SEARCH=0")
            add_test(NAME strix-gemma4-device-ops-tile-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-tile-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_OUTPUT_TILE=0")
            add_test(NAME strix-gemma4-device-ops-tile-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-tile-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_OUTPUT_TILE=1")
            add_test(NAME strix-gemma4-device-rows-publish-off-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-publish-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_KV_PUBLISH=0")
            add_test(NAME strix-gemma4-device-rows-linear-finite-off-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-linear-finite-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_LINEAR_FINITE=0")
            add_test(NAME strix-gemma4-device-ops-norm-prefetch-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-norm-prefetch-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_PREFETCH=0")
            add_test(NAME strix-gemma4-device-rows-norm-finite-off-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-norm-finite-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_FINITE=0")
            add_test(NAME strix-gemma4-device-rows-pinned-io-off-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-pinned-io-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_PINNED_IO=0")
            add_test(NAME strix-gemma4-device-ops-norm-bounds-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-norm-bounds-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_BOUNDS=0")
            add_test(NAME strix-gemma4-device-ops-chunk-prefetch-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-chunk-prefetch-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_CHUNK_PREFETCH=1")
            add_test(NAME strix-gemma4-device-ops-chunk-prefetch-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-chunk-prefetch-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_CHUNK_PREFETCH=0")
            add_test(NAME strix-gemma4-device-rows-chunk-prefetch-on-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-chunk-prefetch-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_CHUNK_PREFETCH=1")
            add_test(NAME strix-gemma4-device-ops-packed-header-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-packed-header-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_PACKED_HEADER=1")
            add_test(NAME strix-gemma4-device-ops-packed-header-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-packed-header-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_PACKED_HEADER=0")
            add_test(NAME strix-gemma4-device-ops-streaming-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-streaming-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_STREAMING_WEIGHTS=1")
            add_test(NAME strix-gemma4-device-ops-streaming-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-streaming-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_STREAMING_WEIGHTS=0")
            add_test(NAME strix-gemma4-device-rows-streaming-on-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-streaming-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_STREAMING_WEIGHTS=1")
            add_test(NAME strix-gemma4-device-ops-norm-stage-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-norm-stage-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_STAGE=1")
            add_test(NAME strix-gemma4-device-ops-norm-stage-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-norm-stage-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_STAGE=0")
            add_test(NAME strix-gemma4-device-rows-norm-stage-on-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-norm-stage-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_STAGE=1")
            add_test(NAME strix-gemma4-device-ops-head-registers-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-head-registers-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_HEAD_REGISTERS=1")
            add_test(NAME strix-gemma4-device-ops-head-registers-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-head-registers-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_HEAD_REGISTERS=0")
            add_test(NAME strix-gemma4-device-ops-head-registers-scalar-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-head-registers-scalar-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_HEAD_REGISTERS=1;STRIX_G4_NORM_SEARCH=0")
            add_test(NAME strix-gemma4-device-ops-input-cache-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-input-cache-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_INPUT_CACHE=1")
            add_test(NAME strix-gemma4-device-ops-input-cache-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-input-cache-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_INPUT_CACHE=0")
            add_test(NAME strix-gemma4-device-ops-attention-prefetch-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-attention-prefetch-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_ATTN_PREFETCH=1")
            add_test(NAME strix-gemma4-device-ops-attention-prefetch-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-attention-prefetch-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_ATTN_PREFETCH=0")
            add_test(NAME strix-gemma4-device-rows-attention-prefetch-on-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-attention-prefetch-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_ATTN_PREFETCH=1")
            add_test(NAME strix-gemma4-device-ops-norm-intervals-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-norm-intervals-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_INTERVALS=1")
            add_test(NAME strix-gemma4-device-ops-norm-intervals-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-norm-intervals-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_INTERVALS=0")
            add_test(NAME strix-gemma4-device-rows-norm-intervals-on-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-norm-intervals-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_NORM_INTERVALS=1")
            add_test(NAME strix-gemma4-device-ops-one-chunk-on-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-one-chunk-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_ONE_CHUNK_ATTN=1")
            add_test(NAME strix-gemma4-device-ops-one-chunk-off-tests COMMAND strix-gemma4-device-ops-tests)
            set_tests_properties(strix-gemma4-device-ops-one-chunk-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_ONE_CHUNK_ATTN=0")
            add_test(NAME strix-gemma4-device-rows-one-chunk-on-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-one-chunk-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_ONE_CHUNK_ATTN=1")
            add_test(NAME strix-gemma4-device-rows-layer-norm-on-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-layer-norm-on-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_LAYER_NORM_FUSION=1")
            add_test(NAME strix-gemma4-device-rows-layer-norm-off-tests COMMAND strix-gemma4-device-rows-tests --short)
            set_tests_properties(strix-gemma4-device-rows-layer-norm-off-tests PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
                ENVIRONMENT "STRIX_G4_LAYER_NORM_FUSION=0")
        endif()
    else()
        message(STATUS "Gemma4 device graph and gates await the manifest/host-reference package")
    endif()
    foreach(t IN LISTS gemma4_b_hip_targets)
        set_target_properties(${t} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee;-fno-gpu-flush-denormals-to-zero>)
        # The shared host interface requests dynamic rounding modes. Device arithmetic is fixed RNE;
        # dynamic modes are unsupported by this ISA. Preserve all strict/no-contraction options.
        get_target_property(g4_sources ${t} SOURCES)
        foreach(source IN LISTS g4_sources)
            if(source MATCHES "\\.hip$")
                set_property(SOURCE ${source} APPEND PROPERTY COMPILE_OPTIONS -fno-rounding-math)
            endif()
        endforeach()
        strix_target_warnings(${t})
    endforeach()
endif()
