# MiniMax H3 block glue (package block): weight loader, host reference forward + sampler (always with STRIX_FAMILY_H3),
# the glue kernels K3-K6 and the single-node device forward driver (with STRIX_ENABLE_HIP_RUNTIME).
add_library(strix_h3_block_ref STATIC src/h3/block/host_ref.cpp)
target_link_libraries(strix_h3_block_ref PUBLIC strix_h3_common)
if(TARGET strix_h3_gemm_ref)
    target_link_libraries(strix_h3_block_ref PUBLIC strix_h3_gemm_ref)
endif()
find_package(Threads REQUIRED)
target_link_libraries(strix_h3_block_ref PUBLIC Threads::Threads)
if(MSVC)
    target_compile_options(strix_h3_block_ref PUBLIC /fp:strict)
endif()
if(BUILD_TESTING)
    # CPU tests: "unit" and "sampler-unit" need no checkpoint; the other modes read H3_FIXTURES / H3_CKPT.
    add_executable(strix-h3-block-ref-tests tests/h3/block/block_ref_tests.cpp tests/h3/block/forward_tests.cpp)
    target_link_libraries(strix-h3-block-ref-tests PRIVATE strix_h3_block_ref)
    target_include_directories(strix-h3-block-ref-tests PRIVATE tests/h3/block)
    strix_target_warnings(strix-h3-block-ref-tests)
    add_test(NAME strix-h3-block-ref-tests COMMAND strix-h3-block-ref-tests unit)
    set_tests_properties(strix-h3-block-ref-tests PROPERTIES TIMEOUT 120 LABELS "h3")
    # fixture modes (G2 pieces, sampler / RNG replay): SKIPPED (77) without H3_FIXTURES / H3_CKPT
    add_test(NAME strix-h3-block-ref-fixtures COMMAND strix-h3-block-ref-tests sampler-unit weights layout block0 block0-fl2va refiner)
    set_tests_properties(strix-h3-block-ref-fixtures PROPERTIES TIMEOUT 1800 LABELS "h3" SKIP_RETURN_CODE 77)
endif()
if(STRIX_ENABLE_HIP_RUNTIME AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/h3/block/glue.hip")
    add_library(strix_h3_block_hip STATIC src/h3/block/glue.hip src/h3/block/device_forward.hip src/h3/block/turbo_accel.hip src/h3/block/sparse_accel.hip
        src/h3/block/lora_host.cpp)
    # the levers sparse K1 variant (defined in h3_50_levers.cmake, after this file: resolved at generate time)
    target_link_libraries(strix_h3_block_hip PUBLIC strix_h3_levers_sparse_attn)
    target_link_libraries(strix_h3_block_hip PUBLIC strix_h3_attn)
    target_link_libraries(strix_h3_block_hip PUBLIC strix_h3_block_ref "${STRIX_HIP_RUNTIME_LIBRARY}")
    if(COMMAND strix_h3_gemm_device_target)
        strix_h3_gemm_device_target(strix_h3_block_hip)
    endif()
    target_include_directories(strix_h3_block_hip PUBLIC "${STRIX_HIP_INCLUDE_DIR}")
    set_target_properties(strix_h3_block_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
    target_compile_options(strix_h3_block_hip PUBLIC $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee;-fno-gpu-flush-denormals-to-zero>)
    if(BUILD_TESTING)
        add_executable(strix-h3-block-gpu-tests tests/h3/block/gpu_tests.cpp)
        set_source_files_properties(tests/h3/block/gpu_tests.cpp PROPERTIES LANGUAGE HIP)
        target_link_libraries(strix-h3-block-gpu-tests PRIVATE strix_h3_block_hip)
        target_include_directories(strix-h3-block-gpu-tests PRIVATE tests/h3/block)
        # GPU tests in CTest only on request (needs a gfx1151 GPU): -DSTRIX_H3_GPU_CTEST=ON, labels gpu / h3-gpu;
        # kernels need no assets, forward / g2flash read H3_FIXTURES + H3_CKPT (else SKIPPED, 77)
        if(STRIX_H3_GPU_CTEST)
            add_test(NAME strix-h3-block-gpu-kernels COMMAND strix-h3-block-gpu-tests kernels)
            add_test(NAME strix-h3-block-gpu-forward COMMAND strix-h3-block-gpu-tests forward g2flash)
            set_tests_properties(strix-h3-block-gpu-kernels strix-h3-block-gpu-forward PROPERTIES TIMEOUT 600 LABELS "gpu;h3;h3-gpu" SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
# Single-node clip runner: prompt -> text encoder -> refiner -> DiT sampler -> video + audio VAE -> raw frames + wav.
if(STRIX_ENABLE_HIP_RUNTIME AND TARGET strix_h3_block_hip AND TARGET strix_h3_te_gpu AND TARGET strix_h3_vae_device)
    # strix-h3-clip-testhooks: the same runner with the failure-injection hooks (H3_CLIP_TEST_HOOKS) for the negative
    # script; the production binary compiles them out.
    foreach(clip strix-h3-clip strix-h3-clip-testhooks)
        add_executable(${clip} tools/h3/block/h3_clip.cpp)
        target_link_libraries(${clip} PRIVATE strix_h3_block_hip strix_h3_te_gpu strix_h3_vae_device)
        # Two-node runner (--dit dist, --world 2): the dist package's device driver (defined in h3_60_dist.cmake, same
        # conditions; a target named here before its definition resolves at generate time).
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/h3/dist/device_dist.hip" AND TARGET strix_dist AND TARGET strix_h3_attn)
            target_link_libraries(${clip} PRIVATE strix_h3_dist_hip)
            target_compile_definitions(${clip} PRIVATE H3_CLIP_DIST=1)
        endif()
    endforeach()
    target_compile_definitions(strix-h3-clip-testhooks PRIVATE H3_CLIP_TEST_HOOKS=1)
    # host self-tests of the clip wrapper / compare scripts (PRECHECK gate, POST status, g3_compare rejects): no GPU, no assets
    if(BUILD_TESTING AND UNIX)
        add_test(NAME strix-h3-block-scripts COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/h3/block/script_selftest.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/h3/block)
        set_tests_properties(strix-h3-block-scripts PROPERTIES TIMEOUT 120 LABELS "h3")
    endif()
    # runner G3 (single == dist world 1 == world 2 emulate, dense + Turbo) and the negative / failure-path script: GPU,
    # assets from H3_MODELS / H3_TOKENIZER_DIR (else SKIPPED, 77)
    if(BUILD_TESTING AND STRIX_H3_GPU_CTEST AND UNIX AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/h3/dist/device_dist.hip"
       AND TARGET strix_dist AND TARGET strix_h3_attn)
        add_test(NAME strix-h3-clip-g3 COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/h3/block/g3_runner_test.sh $<TARGET_FILE:strix-h3-clip>
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/h3/block/g3_compare.sh)
        set_tests_properties(strix-h3-clip-g3 PROPERTIES TIMEOUT 900 LABELS "gpu;h3;h3-gpu" SKIP_RETURN_CODE 77)
        add_test(NAME strix-h3-clip-negative COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/h3/block/runner_negative_test.sh
            $<TARGET_FILE:strix-h3-clip> ${CMAKE_CURRENT_SOURCE_DIR}/tools/h3/block/g3_compare.sh $<TARGET_FILE:strix-h3-clip-testhooks>)
        set_tests_properties(strix-h3-clip-negative PROPERTIES TIMEOUT 900 LABELS "gpu;h3;h3-gpu" SKIP_RETURN_CODE 77)
    endif()
endif()
