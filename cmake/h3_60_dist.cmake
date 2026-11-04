# MiniMax H3 two-node sequence-parallel schedule (package dist, D1): host reference + in-process exchange (always with
# STRIX_FAMILY_H3), the device driver on the engine's communicator and the rank tool (with STRIX_ENABLE_HIP_RUNTIME).
add_library(strix_h3_dist_ref STATIC src/h3/dist/host_dist.cpp src/h3/dist/kv_exchange.cpp)
target_link_libraries(strix_h3_dist_ref PUBLIC strix_h3_block_ref strix_h3_attn_ref)
if(BUILD_TESTING)
    # CPU tests: "unit" needs no checkpoint; "forward" / "mid" read H3_FIXTURES and H3_CKPT.
    add_executable(strix-h3-dist-host-tests tests/h3/dist/dist_host_tests.cpp)
    target_link_libraries(strix-h3-dist-host-tests PRIVATE strix_h3_dist_ref)
    strix_target_warnings(strix-h3-dist-host-tests)
    add_test(NAME strix-h3-dist-host-tests COMMAND strix-h3-dist-host-tests unit)
    set_tests_properties(strix-h3-dist-host-tests PROPERTIES TIMEOUT 300 LABELS "h3" SKIP_RETURN_CODE 77)
endif()
if(STRIX_ENABLE_HIP_RUNTIME AND TARGET strix_dist AND TARGET strix_h3_block_hip AND TARGET strix_h3_attn)
    add_library(strix_h3_dist_hip STATIC src/h3/dist/device_dist.hip)
    target_link_libraries(strix_h3_dist_hip PUBLIC strix_h3_dist_ref strix_h3_block_hip strix_h3_attn strix_h3_gemm_ref strix_dist)
    if(TARGET strix_h3_levers_sparse_attn)  # sparse attention identity (accelerator; levers sparse K1)
        target_link_libraries(strix_h3_dist_hip PUBLIC strix_h3_levers_sparse_attn)
    endif()
    set_target_properties(strix_h3_dist_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
    target_compile_options(strix_h3_dist_hip PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee;-fno-gpu-flush-denormals-to-zero>)
    # Rank tool (one process per rank on the pair, or both ranks in threads: --transport emulate).
    add_executable(strix-h3-dist-forward tools/h3/dist/dist_forward_main.cpp)
    target_link_libraries(strix-h3-dist-forward PRIVATE strix_h3_dist_hip)
endif()
