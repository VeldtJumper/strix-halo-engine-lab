cmake_minimum_required(VERSION 3.24)

project(strix_engine_lab VERSION 0.1.0 LANGUAGES CXX)

# Standalone CUDA Qwen prefill and host-only scheduling/codec checks.
option(STRIX_CUDA_QWEN_ONLY "Build only the CUDA Qwen prefill service and gates" OFF)
if(STRIX_CUDA_QWEN_ONLY)
    include(cmake/QwenCuda.cmake)
    return()
endif()

if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Evaluate after all optional language/toolchain and STRIX option configuration.
cmake_language(DEFER CALL include "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StrixBuildInfo.cmake")

# Optional evaluated-input export only. No manifest is self-generated or
# admitted by configuration; explicit valid Ninja dependency export is required.
option(STRIX_EXPORT_OFFLINE_TUNING_PROVENANCE "Export Ninja POSIX FileAPI inputs for offline compile provenance" OFF)
if(STRIX_EXPORT_OFFLINE_TUNING_PROVENANCE)
    if(NOT UNIX OR NOT CMAKE_GENERATOR STREQUAL "Ninja")
        message(FATAL_ERROR "Offline compile provenance export supports single-config Ninja POSIX only")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/.cmake/api/v1/query")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/codemodel-v2" "")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/toolchains-v1" "")
endif()

option(STRIX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(STRIX_ENABLE_HIP_RUNTIME "Build optional HIP allocation/runtime support" OFF)
option(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT "Use non-temporal weight loads in routed Q2_K decode down projections" OFF)
if(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
    if(NOT STRIX_ENABLE_HIP_RUNTIME)
        message(FATAL_ERROR "MALL Q2_K down pilot requires HIP runtime support")
    endif()
    set_property(SOURCE src/hip_ds41_formats.hip APPEND PROPERTY COMPILE_DEFINITIONS STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT=1)
endif()
option(STRIX_FAMILY_DEEPSEEK41 "Build the deepseek41 model family module" ON)
option(STRIX_FAMILY_MINIMAX "Build the experimental minimax-m3 bring-up modules (MSA host reference)" OFF)
option(STRIX_ENABLE_CUDA_DS41 "Build the deepseek41 CUDA backend (sm_120 / sm_89 / sm_86) and its tests" OFF)
option(STRIX_ENABLE_CUDA_TRELLIS "Build experimental compressed-weight CUDA linear kernels and probes" OFF)
option(STRIX_DS41_TAP_CAPTURE "Build the host-forward capture tool of the deepseek41 CUDA checks (host only)" OFF)
# Minimal configure for the deepseek41 CUDA backend alone (used for the native Windows build): the host
# libraries it links (a trimmed strix_core, the deepseek41 family, ds41 formats/layer/attention, the host
# forward runtime and the prefill plan), the CUDA libraries, their tests and apps. Everything else (HIP,
# serving, networking, pipeline wire, DSpark) is skipped.
option(STRIX_CUDA_DS41_ONLY "Configure only the deepseek41 CUDA backend and the host libraries it needs" OFF)
if(STRIX_CUDA_DS41_ONLY)
    if(STRIX_ENABLE_HIP_RUNTIME OR NOT STRIX_FAMILY_DEEPSEEK41)
        message(FATAL_ERROR "STRIX_CUDA_DS41_ONLY needs STRIX_FAMILY_DEEPSEEK41=ON and STRIX_ENABLE_HIP_RUNTIME=OFF")
    endif()
    set(STRIX_ENABLE_CUDA_DS41 ON CACHE BOOL "Build the deepseek41 CUDA backend (sm_120 / sm_89 / sm_86) and its tests" FORCE)
    if(MSVC)
        # Host arithmetic as on Linux (-ffp-contract=off): /fp:precise without /fp:contract never contracts
        # into FMA (and x64 SSE2 code has no FMA instruction); float evaluates in float.
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:precise> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor> $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    endif()
endif()
if(STRIX_ENABLE_CUDA_TRELLIS AND (NOT STRIX_ENABLE_CUDA_DS41 OR NOT STRIX_FAMILY_DEEPSEEK41))
    message(FATAL_ERROR "STRIX_ENABLE_CUDA_TRELLIS requires STRIX_ENABLE_CUDA_DS41 and STRIX_FAMILY_DEEPSEEK41")
endif()
option(STRIX_EXPERIMENTAL_MALL_SHARED_DOWN_NT "Use non-temporal Q8_0 weight loads in shared-expert dense-A8 decode down projections" OFF)
if(STRIX_EXPERIMENTAL_MALL_SHARED_DOWN_NT)
    if(NOT STRIX_ENABLE_HIP_RUNTIME)
        message(FATAL_ERROR "MALL shared down pilot requires HIP runtime support")
    endif()
    set_property(SOURCE src/hip_shared_expert_launcher.hip APPEND PROPERTY COMPILE_DEFINITIONS STRIX_EXPERIMENTAL_MALL_SHARED_DOWN_NT=1)
endif()
option(STRIX_USE_OPENSSL_SHA256 "Use OpenSSL for full-file SHA-256 verification" OFF)
option(STRIX_CHECK_KERNEL_SCRATCH "Fail the build when a kernel's private scratch/spills exceed tools/kernel_scratch_allowance.txt" OFF)
# Post-build check of every amdgcn code object linked into target (tools/check_kernel_scratch.py).
function(strix_check_kernel_scratch target)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/check_kernel_scratch.py --quiet
            --allowance ${CMAKE_SOURCE_DIR}/tools/kernel_scratch_allowance.txt $<TARGET_FILE:${target}>
        COMMENT "Kernel scratch/spill check: ${target}" VERBATIM)
endfunction()
option(STRIX_ENABLE_EXPERIMENTAL_DENSE_TRELLIS "Build the explicit experimental dense trellis pack route" OFF)
option(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE "Build standalone experimental Q4K A8G32 HIP probe" OFF)
option(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME "Build opt-in native Q4K A8G32 gate/up integration" OFF)
set(STRIX_HC_REPACKED_ADMITTED_SCHEDULES "" CACHE STRING
    "Externally admitted HC research schedules (canonical ascending comma-separated subset of 31,32,33)")
# This only validates syntax, never compiler/ISA or numerical admission. The
# caller must bind the exact nonempty subset to an external qualification receipt.
set(_strix_hc_repacked_canonical_lists "31" "32" "33" "31,32" "31,33" "32,33" "31,32,33")
if(NOT "${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}" STREQUAL "")
    list(FIND _strix_hc_repacked_canonical_lists "${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}" _strix_hc_repacked_list_index)
    if(_strix_hc_repacked_list_index EQUAL -1)
        message(FATAL_ERROR "STRIX_HC_REPACKED_ADMITTED_SCHEDULES must be an ascending comma-separated subset of 31,32,33")
    endif()
endif()
unset(_strix_hc_repacked_canonical_lists)
unset(_strix_hc_repacked_list_index)
if((STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME) AND NOT STRIX_ENABLE_HIP_RUNTIME)
    message(FATAL_ERROR "Experimental Q4K A8 probe requires HIP runtime support")
endif()

function(strix_target_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(STRIX_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(STRIX_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

# Named host-table arithmetic: no reassociation or implicit multiply-add.
if(MSVC)
    set_source_files_properties(src/rotary_descriptor.cpp src/qwen_yarn_table.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
    set_source_files_properties(src/rotary_descriptor.cpp src/qwen_yarn_table.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-frounding-math")
endif()

find_package(Threads REQUIRED)

# Prompt-lookup (n-gram) index shared by the engines' drafting (it only proposes draft tokens: verification decides).
add_library(strix_lookup STATIC src/lookup/index.cpp)
target_include_directories(strix_lookup PUBLIC include)
target_compile_features(strix_lookup PUBLIC cxx_std_20)
strix_target_warnings(strix_lookup)
# Shared drafting layer: context proposer (copy cursor, long proposals), lookup/drafter/serial arbiter, stats line.
add_library(strix_spec STATIC src/spec/drafting.cpp)
target_link_libraries(strix_spec PUBLIC strix_lookup)
strix_target_warnings(strix_spec)

if(STRIX_CUDA_DS41_ONLY)
# Only what the deepseek41 family and the ds41 libraries link: GGUF reader, manifest, binding, content
# hash and the family registry.
add_library(strix_core STATIC
    src/gguf_reader.cpp
    src/model_manifest.cpp
    src/model_runtime_binding.cpp
    src/model_content_hash.cpp
    src/prompt_terminal_diagnostics.cpp
    src/model_family_registry.cpp
    src/cost_table_cache.cpp
)
else()
add_library(strix_core STATIC
    src/rotary_descriptor.cpp
    src/qwen_yarn_table.cpp
    src/final_generation_state_export.cpp
    src/greedy_serving_scheduler.cpp
    src/serving_cohort.cpp
    src/generation_output.cpp
    src/known_prompt_chunks.cpp
    src/known_prefix_prefill.cpp
    src/known_prefix_prefill_runtime.cpp
    src/offline_tuning_cache.cpp
    src/offline_tuning_preparation.cpp
    src/offline_compile_provenance.cpp
    src/uma_owner_inventory.cpp
    src/linux_uma_observation.cpp
    src/preflight.cpp
    src/moe_fork_join_arena.cpp
    src/moe_token_cost_backend.cpp
    src/prepared_execution.cpp
    src/prepared_tensor_views.cpp
    src/prepared_arena_views.cpp
    src/prepared_serial_session.cpp
    src/execution_telemetry.cpp
    src/qsa_input_prefetch.cpp
    src/model_copy_residency_layout.cpp
    src/lane_chain_hyper_output.cpp
    src/shared_expert_prefetch.cpp
    src/q51_grouped_down_contract.cpp
    src/q51_slot_parallel_contract.cpp
    src/moe_q4k_a8_runtime.cpp
    src/gdn_output_admission.cpp
    src/arithmetic_lineage.cpp
    src/session_executor_handoff.cpp
    src/graph_eligibility.cpp
    src/graph_inputs.cpp
    src/graph_attention_state.cpp
    src/graph_ple.cpp
    src/graph_operator_control.cpp
    src/integer_projection.cpp
    src/integer_projection_execution.cpp
    src/integer_projection_runtime.cpp
    src/host_upload.cpp
    src/compiled_execution.cpp
    src/copied_stream_transport.cpp
    src/direct_transport_gpu_export.cpp
    src/direct_gpu_boundary_staging.cpp
    src/linux_direct_transport_driver.cpp
    src/direct_gpu_stream_commissioning.cpp
    src/raw_baseline_recording.cpp
    src/compiled_mtp_draft_plan.cpp
    src/compiled_speculative_verify.cpp
    src/compiled_speculation_backend.cpp
    src/compiled_sampled_speculation_backend.cpp
    src/dispatch_resources.cpp
    src/distributed_flow.cpp
    src/distributed_placement.cpp
    src/distributed_data_plane.cpp
    src/distributed_native_backend.cpp
    src/distributed_token_transaction.cpp
    src/distributed_stage_resources.cpp
    src/distributed_stage_state.cpp
    src/distributed_greedy_control.cpp
    src/distributed_remote_control.cpp
    src/distributed_control_wire.cpp
    src/distributed_control_server.cpp
    src/distributed_control_receiver.cpp
    $<$<PLATFORM_ID:Linux,Darwin>:src/distributed_tcp_control_stream.cpp>
    src/distributed_runtime.cpp
    src/distributed_session.cpp
    src/direct_transport_completion.cpp
    src/direct_transport_driver.cpp
    src/direct_stream_control.cpp
    src/direct_transport_native_backend.cpp
    src/direct_transport_native_session.cpp
    src/direct_transport_frames.cpp
    src/direct_transport_lease.cpp
    src/direct_transport_pool.cpp
    src/direct_transport_pool_memory.cpp
    src/direct_transport_rearm.cpp
    src/dual_link_runtime.cpp
    src/dual_link_telemetry.cpp
    src/draft_proposal_state.cpp
    src/execution_arena.cpp
    src/execution_binding.cpp
    src/execution_buffers.cpp
    src/execution_io.cpp
    src/execution_memory.cpp
    src/execution_plan.cpp
    src/execution_runtime.cpp
    src/execution_tensor_binding.cpp
    src/execution_tensor_access.cpp
    src/execution_resources.cpp
    src/gguf_reader.cpp
    src/model_family_registry.cpp
    src/kernel_registry.cpp
    src/mapped_model_file.cpp
    src/iq4_xs.cpp
    src/indexed_attention_workspace.cpp
    src/indexed_attention_state.cpp
    src/iq4_nl.cpp
    src/ggml_dequant.cpp
    src/q6p_registry.cpp
    src/launch_dispatch.cpp
    src/low_precision.cpp
    src/memory_plan.cpp
    src/model_contract.cpp
    src/model_content_hash.cpp
    src/prompt_terminal_diagnostics.cpp
    src/mtp_draft_binding.cpp
    src/mtp_draft_contract.cpp
    src/mtp_draft_qualification.cpp
    src/mtp_ngram_draft.cpp
    src/mtp_frame_draft.cpp
    src/mtp_context_draft.cpp
    src/mtp_draft_tensor_audit.cpp
    src/mtp_draft_state_contract.cpp
    src/mtp_draft_state_memory.cpp
    src/mtp_draft_readiness.cpp
    src/mtp_draft_activation.cpp
    src/mtp_draft_cycle_input.cpp
    src/mtp_forward_replay.cpp
    src/mtp_forward_contract.cpp
    src/mtp_forward_golden_file.cpp
    src/mtp_target_window_policy.cpp
    src/mtp_selection_transaction.cpp
    src/mtp_window_controller.cpp
    src/cost_table_cache.cpp
    src/mtp_cost_table_session.cpp
    src/mtp_depth_window_controller.cpp
    src/mtp_sidecar_binding.cpp
    src/mtp_forward_semantics.cpp
    src/mtp_forward_runtime.cpp
    src/mtp_input_fusion.cpp
    src/mtp_output_contract.cpp
    src/mtp_attention_contract.cpp
    src/qsa_selection_plan.cpp
    src/qualified_mtp_draft_backend.cpp
    src/qualified_mtp_forward_runtime.cpp
    src/model_manifest.cpp
    src/model_runtime_binding.cpp
    src/model_tensor_index.cpp
    src/moe_plan.cpp
    src/moe_layer_schedule.cpp
    src/moe_layer_runtime.cpp
    src/partition_search.cpp
    src/oracles.cpp
    src/packed_projection.cpp
    src/recurrent_plan.cpp
    src/runtime_allocator.cpp
    src/runtime_launchers.cpp
    src/serial_decode_reference.cpp
    src/serial_decode_replay.cpp
    src/serial_prefix_cache.cpp
    src/serial_prefix_lifecycle.cpp
    src/serial_request.cpp
    src/serial_session.cpp
    src/session_state_layout.cpp
    src/session_state_memory.cpp
    src/session_snapshot.cpp
    src/session_snapshot_memory.cpp
    src/session_snapshot_transfer.cpp
    src/mtp_target_checkpoint_plan.cpp
    src/mtp_draft_vocabulary.cpp
    src/session_restore.cpp
    src/sparse_lookup.cpp
    src/sparse_lookup_executor.cpp
    src/speculation.cpp
    src/speculation_controller.cpp
    src/speculation_measurement.cpp
    src/speculation_rng.cpp
    src/speculation_sampling.cpp
    src/speculation_sampling_cycle.cpp
    src/speculation_sampling_executor.cpp
    src/token_sampling.cpp
    src/speculation_cycle.cpp
    src/speculation_executor.cpp
    src/speculative_verify_plan.cpp
    src/speculative_verify_runtime.cpp
    src/speculative_verify_binding.cpp
    src/platform_probe.cpp
    src/prompt_lookup_draft.cpp
    src/projection_reduction.cpp
    src/ple_lookup.cpp
    src/ple_reclaim.cpp
    src/ple_direct.cpp
    src/ple_rolling.cpp
    src/ple_workspace.cpp
    src/session_state_cache.cpp
    src/prefix_state_cache.cpp
    src/rtx_state_import.cpp
    src/bp_codec.cpp
    src/transport.cpp
)
endif()
target_include_directories(strix_core PUBLIC include)
target_compile_features(strix_core PUBLIC cxx_std_20)
target_link_libraries(strix_core PUBLIC Threads::Threads)
if(STRIX_USE_OPENSSL_SHA256)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    target_link_libraries(strix_core PRIVATE OpenSSL::Crypto)
    target_compile_definitions(strix_core PRIVATE STRIX_USE_OPENSSL_SHA256=1)
endif()
strix_target_warnings(strix_core)

include(CTest)
set(STRIX_TIERSTREAM_DIR "" CACHE PATH "tierstream source tree (empty: no multi-disk reads)")
if(STRIX_TIERSTREAM_DIR)
    if(STRIX_ENABLE_CUDA_DS41)
        set(strix_tierstream_build "build-cuda")
    else()
        set(strix_tierstream_build "build-cpu")
    endif()
    set(STRIX_TIERSTREAM_LIBRARY "${STRIX_TIERSTREAM_DIR}/${strix_tierstream_build}/tierstream${CMAKE_STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "tierstream archive matching this build's backend (may be an out-of-tree build)")
    add_library(strix_tierstream STATIC IMPORTED)
    set_target_properties(strix_tierstream PROPERTIES IMPORTED_LOCATION "${STRIX_TIERSTREAM_LIBRARY}"
                  INTERFACE_INCLUDE_DIRECTORIES "${STRIX_TIERSTREAM_DIR}/include")
    target_link_libraries(strix_tierstream INTERFACE Threads::Threads "$<$<PLATFORM_ID:Windows>:avrt;winmm;bcrypt>")
    file(STRINGS "${STRIX_TIERSTREAM_DIR}/include/strix_stream/ss.h" strix_mfile_define REGEX "^#define SS_MFILE +1")
    if(strix_mfile_define)
        target_compile_definitions(strix_tierstream INTERFACE STRIX_HAVE_TIERSTREAM_MFILE=1)
    endif()
    # Device profiles (ss_devprof_*): the library loads the HIP / CUDA / NVML runtimes at run time.
    file(STRINGS "${STRIX_TIERSTREAM_DIR}/include/strix_stream/ss.h" strix_devprof_define REGEX "^#define SS_DEVPROF +1")
    if(strix_devprof_define)
        target_compile_definitions(strix_tierstream INTERFACE STRIX_HAVE_TIERSTREAM_DEVPROF=1)
        target_link_libraries(strix_tierstream INTERFACE ${CMAKE_DL_LIBS})
    endif()
endif()

if(BUILD_TESTING)
    add_executable(strix-ds41-file-reader-tests tests/ds41_file_reader_multi_tests.cpp src/ds41_file_reader.cpp)
    target_include_directories(strix-ds41-file-reader-tests PRIVATE include)
    target_compile_features(strix-ds41-file-reader-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-ds41-file-reader-tests PRIVATE Threads::Threads)
    if(STRIX_TIERSTREAM_DIR AND strix_mfile_define)
        target_link_libraries(strix-ds41-file-reader-tests PRIVATE strix_tierstream)
        target_include_directories(strix-ds41-file-reader-tests PRIVATE "${STRIX_TIERSTREAM_DIR}/src")
    endif()
    add_test(NAME strix-ds41-file-reader-tests COMMAND strix-ds41-file-reader-tests)
    set_tests_properties(strix-ds41-file-reader-tests PROPERTIES LABELS "cpu;io" TIMEOUT 60)
endif()
if(STRIX_TIERSTREAM_DIR AND strix_mfile_define)
    add_executable(strix-ds41-read-bench apps/ds41_read_bench.cpp src/ds41_file_reader.cpp)
    target_include_directories(strix-ds41-read-bench PRIVATE include)
    target_compile_features(strix-ds41-read-bench PRIVATE cxx_std_20)
    target_link_libraries(strix-ds41-read-bench PRIVATE strix_tierstream Threads::Threads)
    if(BUILD_TESTING)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME strix-ds41-read-tools-tests COMMAND ${Python3_EXECUTABLE} -B
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/ds41_read_tools_tests.py --bench $<TARGET_FILE:strix-ds41-read-bench>)
            set_tests_properties(strix-ds41-read-tools-tests PROPERTIES LABELS "cpu;io" TIMEOUT 60)
        endif()
    endif()
endif()
# Model readers and placement tools need JSON without the POSIX HTTP server.
add_library(strix_serve_json STATIC src/serve/json.cpp)
target_include_directories(strix_serve_json PUBLIC include)
target_compile_features(strix_serve_json PUBLIC cxx_std_20)
strix_target_warnings(strix_serve_json)
target_link_libraries(strix_core PUBLIC strix_serve_json)

if(NOT STRIX_CUDA_DS41_ONLY)
if(NOT MSVC)
    set_source_files_properties(src/low_precision.cpp src/projection_reduction.cpp src/integer_projection.cpp src/ggml_dequant.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
else()
    set_source_files_properties(src/low_precision.cpp src/projection_reduction.cpp src/integer_projection.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
endif()

add_executable(strix-low-precision-reference lab/low_precision/reference_main.cpp)
target_link_libraries(strix-low-precision-reference PRIVATE strix_core)
strix_target_warnings(strix-low-precision-reference)

add_executable(strix-memory-plan apps/memory_plan_main.cpp)
target_link_libraries(strix-memory-plan PRIVATE strix_core)
strix_target_warnings(strix-memory-plan)

add_executable(strix-inspect-gguf apps/inspect_gguf_main.cpp)
target_link_libraries(strix-inspect-gguf PRIVATE strix_core)
strix_target_warnings(strix-inspect-gguf)

# Host-only serving front end (no GPU): JSON, tokenizer, chat template, HTTP and the API mapping.
add_library(strix_serve STATIC
    src/serve/json_constraint.cpp
    src/serve/constraint_selection.cpp
    src/serve/bpe_tokenizer.cpp
    src/serve/chat_template.cpp
    src/serve/deepseek_chat.cpp
    src/serve/glm_chat.cpp
    src/serve/http_server.cpp
    src/serve/engine.cpp
    src/serve/api.cpp
    src/serve/request_fields.cpp
    src/serve/health.cpp
)
target_include_directories(strix_serve PUBLIC include PRIVATE src/serve)
target_compile_features(strix_serve PUBLIC cxx_std_20)
target_link_libraries(strix_serve PUBLIC Threads::Threads strix_core strix_serve_json)
strix_target_warnings(strix_serve)

add_executable(strix-tokenize apps/tokenize_main.cpp)
target_link_libraries(strix-tokenize PRIVATE strix_serve)
strix_target_warnings(strix-tokenize)

add_executable(strix-mtp-manifest-audit apps/mtp_manifest_audit_main.cpp)
target_link_libraries(strix-mtp-manifest-audit PRIVATE strix_core)
strix_target_warnings(strix-mtp-manifest-audit)

add_library(strix_assets STATIC src/asset_manifest.cpp src/serve/json.cpp)
target_link_libraries(strix_assets PUBLIC strix_core)
strix_target_warnings(strix_assets)
add_executable(strix-model-manifest apps/model_manifest_main.cpp)
add_executable(strix-gsq-content-identity lab/gsq/content_identity.cpp)
target_link_libraries(strix-gsq-content-identity PRIVATE strix_core)
target_link_libraries(strix-model-manifest PRIVATE strix_assets)
strix_target_warnings(strix-model-manifest)

add_executable(strix-platform-probe apps/platform_probe_main.cpp)
target_link_libraries(strix-platform-probe PRIVATE strix_core)
strix_target_warnings(strix-platform-probe)

add_executable(strix-validate-serial-reference apps/serial_decode_reference_main.cpp)
target_link_libraries(strix-validate-serial-reference PRIVATE strix_core)
strix_target_warnings(strix-validate-serial-reference)

add_executable(strix-audit-execution-tensors apps/execution_tensor_audit_main.cpp)
target_link_libraries(strix-audit-execution-tensors PRIVATE strix_core)
strix_target_warnings(strix-audit-execution-tensors)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(strix-direct-gpu-observe-driver
        lab/direct_gpu_stream/observe_main.cpp lab/direct_gpu_stream/observe_driver.cpp)
    target_link_libraries(strix-direct-gpu-observe-driver PRIVATE strix_core)
    strix_target_warnings(strix-direct-gpu-observe-driver)
endif()

if(BUILD_TESTING OR STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
    add_library(strix_q4k_a8_contract STATIC src/moe_q4k_a8_contract.cpp src/moe_a8_down_contract.cpp)
    target_include_directories(strix_q4k_a8_contract PUBLIC include)
    target_compile_features(strix_q4k_a8_contract PUBLIC cxx_std_20)
    strix_target_warnings(strix_q4k_a8_contract)
    if(MSVC)
        target_compile_options(strix_q4k_a8_contract PRIVATE /fp:strict)
    else()
        target_compile_options(strix_q4k_a8_contract PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
endif()
if(BUILD_TESTING)
    add_executable(strix-spec-drafting-tests tests/spec_drafting_tests.cpp)
    target_link_libraries(strix-spec-drafting-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-drafting-tests)
    add_test(NAME strix-spec-drafting-tests COMMAND strix-spec-drafting-tests)
    add_executable(strix-batch-nested-recipe-tests tests/batch_nested_recipe_tests.cpp)
    target_include_directories(strix-batch-nested-recipe-tests PRIVATE include)
    target_compile_features(strix-batch-nested-recipe-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-batch-nested-recipe-tests)
    add_test(NAME strix-batch-nested-recipe-tests COMMAND strix-batch-nested-recipe-tests)
    add_executable(strix-batch-service-handoff-tests tests/batch_service_handoff_tests.cpp)
    target_include_directories(strix-batch-service-handoff-tests PRIVATE include)
    target_link_libraries(strix-batch-service-handoff-tests PRIVATE Threads::Threads)
    target_compile_features(strix-batch-service-handoff-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-batch-service-handoff-tests)
    add_test(NAME strix-batch-service-handoff-tests COMMAND strix-batch-service-handoff-tests)
    add_executable(strix-rotary-descriptor-tests tests/rotary_descriptor_tests.cpp)
    target_link_libraries(strix-rotary-descriptor-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-rotary-descriptor-tests)
    add_test(NAME strix-rotary-descriptor-tests COMMAND strix-rotary-descriptor-tests)
    add_executable(strix-qwen-yarn-table-tests tests/qwen_yarn_table_tests.cpp)
    target_link_libraries(strix-qwen-yarn-table-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-yarn-table-tests)
    add_test(NAME strix-qwen-yarn-table-tests COMMAND strix-qwen-yarn-table-tests)
    add_executable(strix-qwen-yarn-memory-tests tests/qwen_yarn_memory_tests.cpp)
    target_compile_features(strix-qwen-yarn-memory-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-qwen-yarn-memory-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-qwen-yarn-memory-tests)
    add_test(NAME strix-qwen-yarn-memory-tests COMMAND strix-qwen-yarn-memory-tests)
    set_tests_properties(strix-qwen-yarn-memory-tests PROPERTIES LABELS "cpu" TIMEOUT 10)
endif()

if(BUILD_TESTING)
    add_executable(strix-serve-health-tests tests/serve_health_test.cpp)
    target_link_libraries(strix-serve-health-tests PRIVATE strix_serve strix_assets)
    target_include_directories(strix-serve-health-tests PRIVATE apps)
    strix_target_warnings(strix-serve-health-tests)
    add_test(NAME strix-serve-health-tests COMMAND strix-serve-health-tests)
    set_tests_properties(strix-serve-health-tests PROPERTIES TIMEOUT 60)
    add_executable(strix-serve-frontend-tests tests/serve_frontend_tests.cpp)
    target_link_libraries(strix-serve-frontend-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-frontend-tests)
    add_test(NAME strix-serve-frontend-tests COMMAND strix-serve-frontend-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/chat_template_cases.json)
    add_executable(strix-glm-tokenizer-tests tests/glm_tokenizer_tests.cpp)
    target_link_libraries(strix-glm-tokenizer-tests PRIVATE strix_serve)
    strix_target_warnings(strix-glm-tokenizer-tests)
    set(STRIX_GLM_TOKENIZER_JSON "" CACHE FILEPATH "GLM-4/5 tokenizer.json (optional; enables the HF fixture checks)")
    add_test(NAME strix-glm-tokenizer-tests COMMAND strix-glm-tokenizer-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/glm_tokenizer_fixtures.json ${STRIX_GLM_TOKENIZER_JSON})
    add_executable(strix-glm-chat-tests tests/glm_chat_tests.cpp)
    target_link_libraries(strix-glm-chat-tests PRIVATE strix_serve)
    strix_target_warnings(strix-glm-chat-tests)
    add_test(NAME strix-glm-chat-tests COMMAND strix-glm-chat-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/glm_chat_cases.json)
    add_executable(strix-serve-deepseek-chat-tests tests/serve_deepseek_chat_tests.cpp)
    target_link_libraries(strix-serve-deepseek-chat-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-deepseek-chat-tests)
    add_test(NAME strix-serve-deepseek-chat-tests COMMAND strix-serve-deepseek-chat-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/deepseek_v41_chat_cases.json)
    add_executable(strix-serve-api-tests tests/serve_api_tests.cpp)
    target_link_libraries(strix-serve-api-tests PRIVATE strix_serve strix_core)
    strix_target_warnings(strix-serve-api-tests)
    add_test(NAME strix-serve-api-tests COMMAND strix-serve-api-tests)

    add_executable(strix-mtp-selection-transaction-tests tests/mtp_selection_transaction_tests.cpp)
    target_link_libraries(strix-mtp-selection-transaction-tests PRIVATE strix_serve)
    strix_target_warnings(strix-mtp-selection-transaction-tests)
    add_test(NAME strix-mtp-selection-transaction-tests COMMAND strix-mtp-selection-transaction-tests)

    add_executable(strix-serve-json-constraint-tests tests/serve_json_constraint_tests.cpp)
    target_link_libraries(strix-serve-json-constraint-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-json-constraint-tests)
    add_test(NAME strix-serve-json-constraint-tests COMMAND strix-serve-json-constraint-tests)
    add_executable(strix-serve-observe-tests tests/serve_observe_tests.cpp)
    target_link_libraries(strix-serve-observe-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-observe-tests)
    add_test(NAME strix-serve-observe-tests COMMAND strix-serve-observe-tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")  # glibc mallinfo2 / malloc_trim for the heap check
        add_executable(strix-serve-observe-cost-tests tests/serve_observe_cost_tests.cpp)
        target_link_libraries(strix-serve-observe-cost-tests PRIVATE strix_serve)
        strix_target_warnings(strix-serve-observe-cost-tests)
        add_test(NAME strix-serve-observe-cost-tests COMMAND strix-serve-observe-cost-tests)
    endif()
    add_executable(strix-teacher-logit-spool-tests tests/teacher_logit_spool_tests.cpp)
    target_compile_features(strix-teacher-logit-spool-tests PRIVATE cxx_std_20)
    target_include_directories(strix-teacher-logit-spool-tests PRIVATE include)
    strix_target_warnings(strix-teacher-logit-spool-tests)
    add_test(NAME strix-teacher-logit-spool-tests COMMAND strix-teacher-logit-spool-tests)
    add_executable(strix-moe-a8-down-contract-tests tests/moe_a8_down_contract_tests.cpp)
    target_link_libraries(strix-moe-a8-down-contract-tests PRIVATE strix_q4k_a8_contract)
    strix_target_warnings(strix-moe-a8-down-contract-tests)
    if(MSVC)
        target_compile_options(strix-moe-a8-down-contract-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-a8-down-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-moe-a8-down-contract-tests COMMAND strix-moe-a8-down-contract-tests)
    add_executable(strix-final-generation-state-export-tests tests/final_generation_state_export_tests.cpp)
    target_link_libraries(strix-final-generation-state-export-tests PRIVATE strix_core)
    strix_target_warnings(strix-final-generation-state-export-tests)
    add_test(NAME strix-final-generation-state-export-tests COMMAND strix-final-generation-state-export-tests)
    add_executable(strix-uma-owner-inventory-tests tests/uma_owner_inventory_tests.cpp)
    target_link_libraries(strix-uma-owner-inventory-tests PRIVATE strix_core)
    strix_target_warnings(strix-uma-owner-inventory-tests)
    add_test(NAME strix-uma-owner-inventory-tests COMMAND strix-uma-owner-inventory-tests)

    add_executable(strix-shared-expert-rows-diagnostic-contract-tests tests/shared_expert_rows_diagnostic_contract_tests.cpp)
    target_link_libraries(strix-shared-expert-rows-diagnostic-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-shared-expert-rows-diagnostic-contract-tests)
    add_test(NAME strix-shared-expert-rows-diagnostic-contract-tests COMMAND strix-shared-expert-rows-diagnostic-contract-tests)

    add_executable(strix-moe-fork-join-arena-tests tests/moe_fork_join_arena_tests.cpp)
    target_link_libraries(strix-moe-fork-join-arena-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-fork-join-arena-tests)
    add_test(NAME strix-moe-fork-join-arena-tests COMMAND strix-moe-fork-join-arena-tests)

    add_executable(strix-prepared-tensor-views-tests tests/prepared_tensor_views_tests.cpp)
    target_link_libraries(strix-prepared-tensor-views-tests PRIVATE strix_core)
    strix_target_warnings(strix-prepared-tensor-views-tests)
    add_test(NAME strix-prepared-tensor-views-tests COMMAND strix-prepared-tensor-views-tests)

    add_executable(strix-prepared-execution-tests tests/prepared_execution_tests.cpp)
    target_link_libraries(strix-prepared-execution-tests PRIVATE strix_core)
    strix_target_warnings(strix-prepared-execution-tests)
    add_test(NAME strix-prepared-execution-tests COMMAND strix-prepared-execution-tests)
    add_executable(strix-generation-output-dispatch-tests tests/generation_output_dispatch_tests.cpp)
    target_link_libraries(strix-generation-output-dispatch-tests PRIVATE strix_core)
    strix_target_warnings(strix-generation-output-dispatch-tests)
    add_test(NAME strix-generation-output-dispatch-tests COMMAND strix-generation-output-dispatch-tests)
    add_executable(strix-greedy-token-request-tests tests/greedy_token_request_tests.cpp)
    target_link_libraries(strix-greedy-token-request-tests PRIVATE strix_core)
    strix_target_warnings(strix-greedy-token-request-tests)
    add_test(NAME strix-greedy-token-request-tests COMMAND strix-greedy-token-request-tests)
    add_executable(strix-execution-inventory-receipt-tests tests/execution_inventory_receipt_tests.cpp)
    target_link_libraries(strix-execution-inventory-receipt-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-inventory-receipt-tests)
    add_test(NAME strix-execution-inventory-receipt-tests COMMAND strix-execution-inventory-receipt-tests)
    add_executable(strix-moe-token-cost-backend-tests tests/moe_token_cost_backend_tests.cpp)
    target_link_libraries(strix-moe-token-cost-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-token-cost-backend-tests)
    add_test(NAME strix-moe-token-cost-backend-tests COMMAND strix-moe-token-cost-backend-tests)

    add_executable(strix-serving-cohort-tests tests/serving_cohort_tests.cpp)
    target_link_libraries(strix-serving-cohort-tests PRIVATE strix_core)
    strix_target_warnings(strix-serving-cohort-tests)
    add_test(NAME strix-serving-cohort-tests COMMAND strix-serving-cohort-tests)

    add_executable(strix-http-session-store-tests tests/http_session_store_tests.cpp)
    target_link_libraries(strix-http-session-store-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-http-session-store-tests)
    add_test(NAME strix-http-session-store-tests COMMAND strix-http-session-store-tests)

    add_executable(strix-session-state-cache-tests tests/session_state_cache_tests.cpp)
    target_link_libraries(strix-session-state-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-state-cache-tests)
    add_test(NAME strix-session-state-cache-tests COMMAND strix-session-state-cache-tests)

    add_executable(strix-qwen-tiled-prompt-plan-tests tests/qwen_tiled_prompt_plan_tests.cpp)
    target_link_libraries(strix-qwen-tiled-prompt-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-tiled-prompt-plan-tests)
    add_test(NAME strix-qwen-tiled-prompt-plan-tests COMMAND strix-qwen-tiled-prompt-plan-tests)

    add_executable(strix-prefix-state-cache-tests tests/prefix_state_cache_tests.cpp tests/prefix_dma_tests.cpp)
    target_include_directories(strix-prefix-state-cache-tests PRIVATE tests/prefix_dma_shim)
    target_link_libraries(strix-prefix-state-cache-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-prefix-state-cache-tests)
    add_test(NAME strix-prefix-state-cache-tests COMMAND strix-prefix-state-cache-tests)
    add_test(NAME strix-prefill-option-cache-write COMMAND strix-prefix-state-cache-tests
        --write-option-checkpoint "${CMAKE_CURRENT_BINARY_DIR}/prefill-option-cache")
    add_test(NAME strix-prefill-option-cache-read COMMAND strix-prefix-state-cache-tests
        --read-option-checkpoint "${CMAKE_CURRENT_BINARY_DIR}/prefill-option-cache")
    set_tests_properties(strix-prefill-option-cache-write PROPERTIES FIXTURES_SETUP prefill_option_cache)
    set_tests_properties(strix-prefill-option-cache-read PROPERTIES FIXTURES_REQUIRED prefill_option_cache)
    add_test(NAME strix-prefill-option-cache-clean COMMAND ${CMAKE_COMMAND} -E rm -rf
        "${CMAKE_CURRENT_BINARY_DIR}/prefill-option-cache")
    set_tests_properties(strix-prefill-option-cache-clean PROPERTIES FIXTURES_CLEANUP prefill_option_cache)

    add_executable(strix-rtx-state-import-tests tests/rtx_state_import_tests.cpp)
    target_include_directories(strix-rtx-state-import-tests PRIVATE apps)
    target_link_libraries(strix-rtx-state-import-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-rtx-state-import-tests)
    add_test(NAME strix-rtx-state-import-tests COMMAND strix-rtx-state-import-tests)
    add_executable(strix-bp-codec-tests tests/bp_codec_tests.cpp)
    target_link_libraries(strix-bp-codec-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-bp-codec-tests)
    add_test(NAME strix-bp-codec-tests COMMAND strix-bp-codec-tests)

    add_executable(strix-known-prompt-chunks-tests tests/known_prompt_chunks_tests.cpp)
    target_link_libraries(strix-known-prompt-chunks-tests PRIVATE strix_core)
    strix_target_warnings(strix-known-prompt-chunks-tests)
    add_test(NAME strix-known-prompt-chunks-tests COMMAND strix-known-prompt-chunks-tests)
    add_executable(strix-serial-prompt-plan-tests tests/serial_prompt_plan_tests.cpp)
    target_link_libraries(strix-serial-prompt-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-prompt-plan-tests)
    foreach(schedule IN ITEMS 2 4 16 20)
        add_test(NAME strix-serial-prompt-plan-sched-${schedule} COMMAND strix-serial-prompt-plan-tests ${schedule})
    endforeach()
    add_executable(strix-http-slot-capacity-tests tests/http_slot_capacity_tests.cpp)
    target_link_libraries(strix-http-slot-capacity-tests PRIVATE strix_core)
    strix_target_warnings(strix-http-slot-capacity-tests)
    add_test(NAME strix-http-slot-capacity-tests COMMAND strix-http-slot-capacity-tests)
    add_executable(strix-qwen-yarn-schedule-tests tests/qwen_yarn_schedule_tests.cpp)
    target_compile_definitions(strix-qwen-yarn-schedule-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    target_link_libraries(strix-qwen-yarn-schedule-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-yarn-schedule-tests)
    foreach(schedule 0 8 13 16 12)
        add_test(NAME strix-qwen-yarn-schedule-${schedule} COMMAND strix-qwen-yarn-schedule-tests)
        set_tests_properties(strix-qwen-yarn-schedule-${schedule} PROPERTIES ENVIRONMENT "STRIX_QPF2K_SCHED=${schedule}")
    endforeach()

    add_executable(strix-hip-prefill-pool-host-tests tests/hip_prefill_pool_host_tests.cpp)
    target_include_directories(strix-hip-prefill-pool-host-tests PRIVATE tests/stubs/hip_prefill_pool src include)
    target_link_libraries(strix-hip-prefill-pool-host-tests PRIVATE Threads::Threads)
    target_compile_features(strix-hip-prefill-pool-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hip-prefill-pool-host-tests)
    add_test(NAME strix-hip-prefill-pool-host-tests COMMAND strix-hip-prefill-pool-host-tests)
    add_executable(strix-hip-prefill-call-shapes-host-tests tests/hip_prefill_call_shapes_host_tests.cpp)
    target_include_directories(strix-hip-prefill-call-shapes-host-tests PRIVATE tests/stubs/hip_prefill_pool src apps)
    target_link_libraries(strix-hip-prefill-call-shapes-host-tests PRIVATE strix_core)
    strix_target_warnings(strix-hip-prefill-call-shapes-host-tests)
    foreach(schedule IN ITEMS 0 12 28)
        add_test(NAME strix-hip-prefill-call-shapes-${schedule}
            COMMAND strix-hip-prefill-call-shapes-host-tests ${schedule})
    endforeach()
    add_executable(strix-qwen-startup-policy-tests tests/qwen_startup_policy_tests.cpp)
    target_link_libraries(strix-qwen-startup-policy-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-qwen-startup-policy-tests)
    add_test(NAME strix-qwen-startup-policy-tests COMMAND strix-qwen-startup-policy-tests)
    add_executable(strix-qwen-effective-config-tests tests/qwen_effective_config_tests.cpp)
    add_executable(strix-qwen-mtp-bundle-tests tests/qwen_mtp_bundle_tests.cpp)
    target_link_libraries(strix-qwen-mtp-bundle-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-mtp-bundle-tests)
    add_test(NAME strix-qwen-mtp-bundle-tests COMMAND strix-qwen-mtp-bundle-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/recipes/qwen-distill-g-yarn.mtp-bundle.json)
    set_tests_properties(strix-qwen-mtp-bundle-tests PROPERTIES LABELS "cpu")
    target_compile_definitions(strix-qwen-effective-config-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1
        STRIX_QWEN_CACHE_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen_cache_public_main_reference.txt")
    target_link_libraries(strix-qwen-effective-config-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-qwen-effective-config-tests)
    add_test(NAME strix-qwen-effective-config-tests COMMAND strix-qwen-effective-config-tests)
    add_test(NAME strix-qwen-yarn-owner-inventory-tests COMMAND strix-qwen-effective-config-tests --yarn-owner-inventory)
    add_test(NAME strix-qwen-serve-prefill-admission-tests COMMAND strix-qwen-effective-config-tests --serve-prefill-admission)
    add_executable(strix-compact-stage-images-tests tests/compact_stage_images_tests.cpp)
    target_link_libraries(strix-compact-stage-images-tests PRIVATE strix_core)
    strix_target_warnings(strix-compact-stage-images-tests)
    add_test(NAME strix-compact-stage-images-tests COMMAND strix-compact-stage-images-tests)
    add_executable(strix-qpf2k-defaults-tests tests/qpf2k_defaults_tests.cpp)
    target_include_directories(strix-qpf2k-defaults-tests PRIVATE apps)
    target_link_libraries(strix-qpf2k-defaults-tests PRIVATE strix_core)
    strix_target_warnings(strix-qpf2k-defaults-tests)
    add_test(NAME strix-qpf2k-defaults-tests COMMAND strix-qpf2k-defaults-tests)
    add_test(NAME strix-qpf2k-readers-gsq-first COMMAND strix-qpf2k-defaults-tests --gsq-first)
    add_test(NAME strix-qpf2k-readers-ordinary-first COMMAND strix-qpf2k-defaults-tests --ordinary-first)
    add_executable(strix-offline-compile-provenance-tests tests/offline_compile_provenance_tests.cpp)
    target_link_libraries(strix-offline-compile-provenance-tests PRIVATE strix_core)
    strix_target_warnings(strix-offline-compile-provenance-tests)
    add_test(NAME offline-compile-provenance COMMAND strix-offline-compile-provenance-tests)
    add_executable(strix-offline-tuning-preparation-tests tests/offline_tuning_preparation_tests.cpp)
    target_link_libraries(strix-offline-tuning-preparation-tests PRIVATE strix_core)
    strix_target_warnings(strix-offline-tuning-preparation-tests)
    add_test(NAME strix-offline-tuning-preparation-tests COMMAND strix-offline-tuning-preparation-tests)
    add_executable(strix-offline-tuning-cache-tests tests/offline_tuning_cache_tests.cpp)
    target_link_libraries(strix-offline-tuning-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-offline-tuning-cache-tests)
    add_test(NAME strix-offline-tuning-cache-tests COMMAND strix-offline-tuning-cache-tests)
    if(UNIX)
        add_executable(strix-startup-markers-tests tests/startup_markers_tests.cpp)
        target_link_libraries(strix-startup-markers-tests PRIVATE strix_core)
        strix_target_warnings(strix-startup-markers-tests)
        add_test(NAME strix-startup-markers-tests COMMAND strix-startup-markers-tests)
    endif()
    add_executable(strix-generation-output-tests tests/generation_output_tests.cpp)
    target_link_libraries(strix-generation-output-tests PRIVATE strix_core)
    strix_target_warnings(strix-generation-output-tests)
    add_test(NAME strix-generation-output-tests COMMAND strix-generation-output-tests)
    add_executable(strix-greedy-serving-scheduler-tests tests/greedy_serving_scheduler_tests.cpp)
    target_link_libraries(strix-greedy-serving-scheduler-tests PRIVATE strix_core)
    strix_target_warnings(strix-greedy-serving-scheduler-tests)
    add_test(NAME strix-greedy-serving-scheduler-tests COMMAND strix-greedy-serving-scheduler-tests)
    add_executable(strix-execution-telemetry-tests tests/execution_telemetry_tests.cpp)
    target_link_libraries(strix-execution-telemetry-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-telemetry-tests)
    add_test(NAME strix-execution-telemetry-tests COMMAND strix-execution-telemetry-tests)
    # Diagnostic-only byte permutation. It is never linked into the decoder.
    add_library(strix_hc_repacked_down_pack STATIC src/hc_repacked_down_pack.cpp)
    target_include_directories(strix_hc_repacked_down_pack PUBLIC include)
    target_compile_features(strix_hc_repacked_down_pack PUBLIC cxx_std_20)
    target_link_libraries(strix_hc_repacked_down_pack PUBLIC strix_core)
    strix_target_warnings(strix_hc_repacked_down_pack)
    add_executable(strix-hc-repacked-down-pack-tests tests/hc_repacked_down_pack_tests.cpp)
    target_link_libraries(strix-hc-repacked-down-pack-tests PRIVATE strix_hc_repacked_down_pack)
    strix_target_warnings(strix-hc-repacked-down-pack-tests)
    add_test(NAME strix-hc-repacked-down-pack-tests COMMAND strix-hc-repacked-down-pack-tests)

    add_executable(strix-q8-projection-organization-tests tests/q8_projection_organization_tests.cpp)
    target_link_libraries(strix-q8-projection-organization-tests PRIVATE strix_core)
    strix_target_warnings(strix-q8-projection-organization-tests)
    add_test(NAME strix-q8-projection-organization-tests COMMAND strix-q8-projection-organization-tests)
    add_executable(strix-hc-projection-prefetch-tests tests/hc_projection_prefetch_tests.cpp)
    target_link_libraries(strix-hc-projection-prefetch-tests PRIVATE strix_core)
    strix_target_warnings(strix-hc-projection-prefetch-tests)
    add_test(NAME strix-hc-projection-prefetch-tests COMMAND strix-hc-projection-prefetch-tests)
    add_executable(strix-q8-projection-options-tests tests/q8_projection_options_tests.cpp)
    target_link_libraries(strix-q8-projection-options-tests PRIVATE strix_core)
    strix_target_warnings(strix-q8-projection-options-tests)
    add_test(NAME strix-q8-projection-options-tests COMMAND strix-q8-projection-options-tests)
    add_executable(strix-gdn-lm-wave8-diagnostic-contract-tests tests/gdn_lm_wave8_diagnostic_contract_tests.cpp)
    target_link_libraries(strix-gdn-lm-wave8-diagnostic-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-gdn-lm-wave8-diagnostic-contract-tests)
    add_test(NAME strix-gdn-lm-wave8-diagnostic-contract-tests COMMAND strix-gdn-lm-wave8-diagnostic-contract-tests)
    add_executable(strix-decode-optimization-profile-tests tests/decode_optimization_profile_tests.cpp)
    add_executable(strix-qwen-product-profile-tests tests/qwen_product_profile_tests.cpp)
    target_link_libraries(strix-qwen-product-profile-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-qwen-product-profile-tests)
    add_test(NAME strix-qwen-product-profile-tests COMMAND strix-qwen-product-profile-tests)
    target_link_libraries(strix-decode-optimization-profile-tests PRIVATE strix_core)
    strix_target_warnings(strix-decode-optimization-profile-tests)
    add_test(NAME strix-decode-optimization-profile-tests COMMAND strix-decode-optimization-profile-tests)
    add_executable(strix-decode-optimization-profile-a8-cli-tests tests/decode_optimization_profile_tests.cpp)
    target_compile_definitions(strix-decode-optimization-profile-a8-cli-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    target_link_libraries(strix-decode-optimization-profile-a8-cli-tests PRIVATE strix_core)
    strix_target_warnings(strix-decode-optimization-profile-a8-cli-tests)
    add_test(NAME strix-decode-optimization-profile-a8-cli-tests COMMAND strix-decode-optimization-profile-a8-cli-tests)
    add_executable(strix-spec-v1-integration-tests tests/spec_v1_integration_tests.cpp)
    target_link_libraries(strix-spec-v1-integration-tests PRIVATE strix_core)
    strix_target_warnings(strix-spec-v1-integration-tests)
    add_test(NAME strix-spec-v1-integration-tests COMMAND strix-spec-v1-integration-tests)
    add_executable(strix-qsa-input-prefetch-tests tests/qsa_input_prefetch_tests.cpp)
    target_link_libraries(strix-qsa-input-prefetch-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-input-prefetch-tests)
    add_test(NAME strix-qsa-input-prefetch-tests COMMAND strix-qsa-input-prefetch-tests)
    add_executable(strix-shared-expert-prefetch-tests tests/shared_expert_prefetch_tests.cpp)
    target_link_libraries(strix-shared-expert-prefetch-tests PRIVATE strix_core)
    strix_target_warnings(strix-shared-expert-prefetch-tests)
    add_test(NAME strix-shared-expert-prefetch-tests COMMAND strix-shared-expert-prefetch-tests)
    add_executable(strix-moe-q4k-a8-contract-tests tests/moe_q4k_a8_contract_tests.cpp)
    target_link_libraries(strix-moe-q4k-a8-contract-tests PRIVATE strix_q4k_a8_contract strix_core)
    strix_target_warnings(strix-moe-q4k-a8-contract-tests)
    if(MSVC)
        target_compile_options(strix-moe-q4k-a8-contract-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-q4k-a8-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-moe-q4k-a8-contract-tests COMMAND strix-moe-q4k-a8-contract-tests)
    add_executable(strix-moe-q5k-a8-contract-tests tests/moe_q5k_a8_contract_tests.cpp)
    target_link_libraries(strix-moe-q5k-a8-contract-tests PRIVATE strix_q4k_a8_contract strix_core)
    strix_target_warnings(strix-moe-q5k-a8-contract-tests)
    if(MSVC)
        target_compile_options(strix-moe-q5k-a8-contract-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-q5k-a8-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-moe-q5k-a8-contract-tests COMMAND strix-moe-q5k-a8-contract-tests)
    add_executable(strix-hc-parallel-projection-oracle-tests tests/hc_parallel_projection_oracle_tests.cpp)
    target_compile_features(strix-hc-parallel-projection-oracle-tests PRIVATE cxx_std_20)
    target_include_directories(strix-hc-parallel-projection-oracle-tests PRIVATE include)
    strix_target_warnings(strix-hc-parallel-projection-oracle-tests)
    add_test(NAME strix-hc-parallel-projection-oracle-tests COMMAND strix-hc-parallel-projection-oracle-tests)
    add_executable(strix-hc-deferred-norm-oracle-tests tests/hc_deferred_norm_oracle_tests.cpp)
    target_link_libraries(strix-hc-deferred-norm-oracle-tests PRIVATE strix_q4k_a8_contract)
    strix_target_warnings(strix-hc-deferred-norm-oracle-tests)
    if(MSVC)
        target_compile_options(strix-hc-deferred-norm-oracle-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-hc-deferred-norm-oracle-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-hc-deferred-norm-oracle-tests COMMAND strix-hc-deferred-norm-oracle-tests)

    add_executable(strix-lane-chain-hyper-output-tests tests/lane_chain_hyper_output_tests.cpp)
    target_link_libraries(strix-lane-chain-hyper-output-tests PRIVATE strix_core)
    strix_target_warnings(strix-lane-chain-hyper-output-tests)
    add_test(NAME strix-lane-chain-hyper-output-tests COMMAND strix-lane-chain-hyper-output-tests)
    add_executable(strix-model-selective-copy-layout-tests tests/model_selective_copy_layout_tests.cpp)
    target_link_libraries(strix-model-selective-copy-layout-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-selective-copy-layout-tests)
    add_test(NAME strix-model-selective-copy-layout-tests COMMAND strix-model-selective-copy-layout-tests)
    add_executable(strix-gdn-output-admission-tests tests/gdn_output_admission_tests.cpp)
    target_link_libraries(strix-gdn-output-admission-tests PRIVATE strix_core)
    strix_target_warnings(strix-gdn-output-admission-tests)
    add_test(NAME strix-gdn-output-admission-tests COMMAND strix-gdn-output-admission-tests)
    add_executable(strix-virtual-q8-gdn-oracle-tests tests/virtual_q8_gdn_oracle_tests.cpp)
    target_compile_features(strix-virtual-q8-gdn-oracle-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-virtual-q8-gdn-oracle-tests)
    add_test(NAME strix-virtual-q8-gdn-oracle-tests COMMAND strix-virtual-q8-gdn-oracle-tests)
    add_executable(strix-router-ranked-topk-tests tests/router_ranked_topk_tests.cpp)
    strix_target_warnings(strix-router-ranked-topk-tests)
    target_compile_features(strix-router-ranked-topk-tests PRIVATE cxx_std_20)
    add_test(NAME strix-router-ranked-topk-tests COMMAND strix-router-ranked-topk-tests)
    add_executable(strix-router-wave-topk-tests tests/router_wave_topk_tests.cpp)
    target_link_libraries(strix-router-wave-topk-tests PRIVATE strix_core)
    target_include_directories(strix-router-wave-topk-tests PRIVATE include)
    target_compile_features(strix-router-wave-topk-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-router-wave-topk-tests)
    add_test(NAME strix-router-wave-topk-tests COMMAND strix-router-wave-topk-tests)
    add_executable(strix-deferred-moe-status-tests tests/deferred_moe_status_tests.cpp)
    target_link_libraries(strix-deferred-moe-status-tests PRIVATE strix_core)
    target_compile_features(strix-deferred-moe-status-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-deferred-moe-status-tests)
    add_test(NAME strix-deferred-moe-status-tests COMMAND strix-deferred-moe-status-tests)
    add_executable(strix-virtual-q8-oracle-tests tests/virtual_q8_oracle_tests.cpp)
    target_compile_features(strix-virtual-q8-oracle-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-virtual-q8-oracle-tests)
    add_test(NAME strix-virtual-q8-oracle-tests COMMAND strix-virtual-q8-oracle-tests)
    add_executable(strix-direct-gpu-stream-commissioning-tests
        tests/direct_gpu_stream_commissioning_tests.cpp
        lab/direct_gpu_stream/bounded_tcp_control.cpp)
    target_link_libraries(strix-direct-gpu-stream-commissioning-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-gpu-stream-commissioning-tests)
    add_test(NAME strix-direct-gpu-stream-commissioning-tests COMMAND strix-direct-gpu-stream-commissioning-tests)
    add_executable(strix-raw-baseline-recording-tests tests/raw_baseline_recording_tests.cpp)
    target_link_libraries(strix-raw-baseline-recording-tests PRIVATE strix_core)
    strix_target_warnings(strix-raw-baseline-recording-tests)
    add_test(NAME strix-raw-baseline-recording-tests COMMAND strix-raw-baseline-recording-tests)
    add_executable(strix-direct-transport-gpu-export-tests tests/direct_transport_gpu_export_tests.cpp)
    target_link_libraries(strix-direct-transport-gpu-export-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-gpu-export-tests)
    add_test(NAME strix-direct-transport-gpu-export-tests COMMAND strix-direct-transport-gpu-export-tests)
    add_executable(strix-graph-ple-tests tests/graph_ple_tests.cpp)
    add_executable(strix-direct-gpu-boundary-staging-tests tests/direct_gpu_boundary_staging_tests.cpp)
    target_link_libraries(strix-direct-gpu-boundary-staging-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-gpu-boundary-staging-tests)
    add_test(NAME strix-direct-gpu-boundary-staging-tests COMMAND strix-direct-gpu-boundary-staging-tests)
    target_link_libraries(strix-graph-ple-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-ple-tests)
    add_test(NAME strix-graph-ple-tests COMMAND strix-graph-ple-tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(strix-direct-gpu-stream-driver-observation-tests
            tests/direct_gpu_stream_driver_observation_tests.cpp lab/direct_gpu_stream/observe_driver.cpp)
        target_link_libraries(strix-direct-gpu-stream-driver-observation-tests PRIVATE strix_core)
        strix_target_warnings(strix-direct-gpu-stream-driver-observation-tests)
        add_test(NAME strix-direct-gpu-stream-driver-observation-tests COMMAND strix-direct-gpu-stream-driver-observation-tests)
        enable_language(C)
        add_executable(strix-linux-direct-transport-driver-tests tests/linux_direct_transport_driver_tests.cpp)
        target_link_libraries(strix-linux-direct-transport-driver-tests PRIVATE strix_core)
        strix_target_warnings(strix-linux-direct-transport-driver-tests)
        add_test(NAME strix-linux-direct-transport-driver-tests COMMAND strix-linux-direct-transport-driver-tests)
        add_executable(strix-linux-direct-gpu-stream-logic-tests lab/direct_gpu_stream/kernel/logic_tests.c)
        target_include_directories(strix-linux-direct-gpu-stream-logic-tests PRIVATE include lab/direct_gpu_stream/kernel)
        target_compile_features(strix-linux-direct-gpu-stream-logic-tests PRIVATE c_std_11)
        strix_target_warnings(strix-linux-direct-gpu-stream-logic-tests)
        add_test(NAME strix-linux-direct-gpu-stream-logic-tests COMMAND strix-linux-direct-gpu-stream-logic-tests)
        add_executable(strix-linux-direct-gpu-stream-uapi-tests tests/linux_direct_gpu_stream_uapi_tests.cpp)
        target_include_directories(strix-linux-direct-gpu-stream-uapi-tests PRIVATE include)
        target_compile_features(strix-linux-direct-gpu-stream-uapi-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-linux-direct-gpu-stream-uapi-tests)
        add_test(NAME strix-linux-direct-gpu-stream-uapi-tests COMMAND strix-linux-direct-gpu-stream-uapi-tests)
    endif()
    add_executable(strix-integer-projection-runtime-tests tests/integer_projection_runtime_tests.cpp)
    target_link_libraries(strix-integer-projection-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-integer-projection-runtime-tests)
    add_test(NAME strix-integer-projection-runtime-tests COMMAND strix-integer-projection-runtime-tests)
    add_executable(strix-graph-operator-control-tests tests/graph_operator_control_tests.cpp)
    target_link_libraries(strix-graph-operator-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-operator-control-tests)
    add_test(NAME strix-graph-operator-control-tests COMMAND strix-graph-operator-control-tests)
    add_executable(strix-copied-stream-transport-tests tests/copied_stream_transport_tests.cpp)
    target_link_libraries(strix-copied-stream-transport-tests PRIVATE strix_core)
    strix_target_warnings(strix-copied-stream-transport-tests)
    add_test(NAME strix-copied-stream-transport-tests COMMAND strix-copied-stream-transport-tests)
    add_executable(strix-integer-projection-execution-tests tests/integer_projection_execution_tests.cpp)
    target_link_libraries(strix-integer-projection-execution-tests PRIVATE strix_core)
    strix_target_warnings(strix-integer-projection-execution-tests)
    add_test(NAME strix-integer-projection-execution-tests COMMAND strix-integer-projection-execution-tests)
    add_executable(strix-graph-attention-state-tests tests/graph_attention_state_tests.cpp)
    target_link_libraries(strix-graph-attention-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-attention-state-tests)
    add_test(NAME strix-graph-attention-state-tests COMMAND strix-graph-attention-state-tests)
    add_executable(strix-direct-stream-control-tests tests/direct_stream_control_tests.cpp)
    target_link_libraries(strix-direct-stream-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-stream-control-tests)
    add_test(NAME strix-direct-stream-control-tests COMMAND strix-direct-stream-control-tests)
    add_executable(strix-direct-transport-driver-tests tests/direct_transport_driver_tests.cpp)
    target_link_libraries(strix-direct-transport-driver-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-driver-tests)
    add_test(NAME strix-direct-transport-driver-tests COMMAND strix-direct-transport-driver-tests)
    add_executable(strix-integer-projection-tests tests/integer_projection_tests.cpp)
    target_link_libraries(strix-integer-projection-tests PRIVATE strix_core)
    strix_target_warnings(strix-integer-projection-tests)
    if(NOT MSVC)
        target_compile_options(strix-integer-projection-tests PRIVATE -ffp-contract=off)
    else()
        target_compile_options(strix-integer-projection-tests PRIVATE /fp:strict)
    endif()
    add_test(NAME strix-integer-projection-tests COMMAND strix-integer-projection-tests)
    add_executable(strix-decode-execution-options-tests tests/decode_execution_options_tests.cpp)
    target_link_libraries(strix-decode-execution-options-tests PRIVATE strix_core)
    strix_target_warnings(strix-decode-execution-options-tests)
    add_test(NAME strix-decode-execution-options-tests COMMAND strix-decode-execution-options-tests)
    add_executable(strix-native-setup-quiescence-tests tests/native_setup_quiescence_tests.cpp)
    target_link_libraries(strix-native-setup-quiescence-tests PRIVATE strix_core)
    strix_target_warnings(strix-native-setup-quiescence-tests)
    add_test(NAME strix-native-setup-quiescence-tests COMMAND strix-native-setup-quiescence-tests)
    add_executable(strix-graph-inputs-tests tests/graph_inputs_tests.cpp)
    target_link_libraries(strix-graph-inputs-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-inputs-tests)
    add_test(NAME strix-graph-inputs-tests COMMAND strix-graph-inputs-tests)
    add_executable(strix-host-upload-tests tests/host_upload_tests.cpp)
    target_link_libraries(strix-host-upload-tests PRIVATE strix_core)
    strix_target_warnings(strix-host-upload-tests)
    add_test(NAME strix-host-upload-tests COMMAND strix-host-upload-tests)
    add_executable(strix-graph-eligibility-tests tests/graph_eligibility_tests.cpp)
    target_link_libraries(strix-graph-eligibility-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-eligibility-tests)
    add_test(NAME strix-graph-eligibility-tests COMMAND strix-graph-eligibility-tests)
    add_executable(strix-projection-reduction-tests tests/projection_reduction_tests.cpp)
    target_link_libraries(strix-projection-reduction-tests PRIVATE strix_core)
    strix_target_warnings(strix-projection-reduction-tests)
    if(NOT MSVC)
        target_compile_options(strix-projection-reduction-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-projection-reduction-tests COMMAND strix-projection-reduction-tests)
    add_executable(strix-low-precision-tests tests/low_precision_tests.cpp)
    target_link_libraries(strix-low-precision-tests PRIVATE strix_core)
    strix_target_warnings(strix-low-precision-tests)
    if(NOT MSVC)
        target_compile_options(strix-low-precision-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-low-precision-tests COMMAND strix-low-precision-tests)
    add_test(NAME strix-low-precision-vectors COMMAND strix-low-precision-reference vectors)
    add_executable(strix-model-family-registry-tests tests/model_family_registry_tests.cpp)
    target_link_libraries(strix-model-family-registry-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-family-registry-tests)
    add_test(NAME strix-model-family-registry-tests COMMAND strix-model-family-registry-tests)
    add_executable(strix-core-tests tests/core_tests.cpp)
    target_link_libraries(strix-core-tests PRIVATE strix_core)
    strix_target_warnings(strix-core-tests)
    add_test(NAME strix-core-tests COMMAND strix-core-tests)

    add_executable(strix-manifest-platform-tests tests/manifest_platform_tests.cpp)
    target_link_libraries(strix-manifest-platform-tests PRIVATE strix_core)
    strix_target_warnings(strix-manifest-platform-tests)
    add_test(NAME strix-manifest-platform-tests COMMAND strix-manifest-platform-tests)

    add_executable(strix-model-runtime-binding-tests tests/model_runtime_binding_tests.cpp)
    target_link_libraries(strix-model-runtime-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-runtime-binding-tests)
    add_test(NAME strix-model-runtime-binding-tests COMMAND strix-model-runtime-binding-tests)

    add_executable(strix-model-content-hash-tests tests/model_content_hash_tests.cpp)
    target_link_libraries(strix-model-content-hash-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-content-hash-tests)
    add_test(NAME strix-model-content-hash-tests COMMAND strix-model-content-hash-tests)

    add_executable(strix-mapped-model-file-tests tests/mapped_model_file_tests.cpp)
    target_link_libraries(strix-mapped-model-file-tests PRIVATE strix_core)
    strix_target_warnings(strix-mapped-model-file-tests)
    add_test(NAME strix-mapped-model-file-tests COMMAND strix-mapped-model-file-tests)

    add_executable(strix-model-tensor-index-tests tests/model_tensor_index_tests.cpp)
    target_link_libraries(strix-model-tensor-index-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-tensor-index-tests)
    add_test(NAME strix-model-tensor-index-tests COMMAND strix-model-tensor-index-tests)

    add_executable(strix-transport-tests tests/transport_tests.cpp)
    target_link_libraries(strix-transport-tests PRIVATE strix_core)
    strix_target_warnings(strix-transport-tests)
    add_test(NAME strix-transport-tests COMMAND strix-transport-tests)

    add_executable(strix-direct-transport-frame-tests tests/direct_transport_frames_tests.cpp)
    target_link_libraries(strix-direct-transport-frame-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-frame-tests)
    add_test(NAME strix-direct-transport-frame-tests COMMAND strix-direct-transport-frame-tests)

    add_executable(strix-direct-transport-pool-tests tests/direct_transport_pool_tests.cpp)
    target_link_libraries(strix-direct-transport-pool-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-pool-tests)
    add_test(NAME strix-direct-transport-pool-tests COMMAND strix-direct-transport-pool-tests)

    add_executable(strix-direct-transport-pool-memory-tests tests/direct_transport_pool_memory_tests.cpp)
    target_link_libraries(strix-direct-transport-pool-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-pool-memory-tests)
    add_test(NAME strix-direct-transport-pool-memory-tests COMMAND strix-direct-transport-pool-memory-tests)

    add_executable(strix-direct-transport-lease-tests tests/direct_transport_lease_tests.cpp)
    target_link_libraries(strix-direct-transport-lease-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-lease-tests)
    add_test(NAME strix-direct-transport-lease-tests COMMAND strix-direct-transport-lease-tests)

    add_executable(strix-direct-transport-completion-tests tests/direct_transport_completion_tests.cpp)
    target_link_libraries(strix-direct-transport-completion-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-completion-tests)
    add_test(NAME strix-direct-transport-completion-tests COMMAND strix-direct-transport-completion-tests)

    add_executable(strix-direct-transport-native-backend-tests tests/direct_transport_native_backend_tests.cpp)
    target_link_libraries(strix-direct-transport-native-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-native-backend-tests)
    add_test(NAME strix-direct-transport-native-backend-tests COMMAND strix-direct-transport-native-backend-tests)

    add_executable(strix-direct-transport-native-session-tests tests/direct_transport_native_session_tests.cpp)
    target_link_libraries(strix-direct-transport-native-session-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-native-session-tests)
    add_test(NAME strix-direct-transport-native-session-tests COMMAND strix-direct-transport-native-session-tests)

    add_executable(strix-direct-transport-rearm-tests tests/direct_transport_rearm_tests.cpp)
    target_link_libraries(strix-direct-transport-rearm-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-rearm-tests)
    add_test(NAME strix-direct-transport-rearm-tests COMMAND strix-direct-transport-rearm-tests)

    add_executable(strix-dual-link-runtime-tests tests/dual_link_runtime_tests.cpp)
    target_link_libraries(strix-dual-link-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-dual-link-runtime-tests)
    add_test(NAME strix-dual-link-runtime-tests COMMAND strix-dual-link-runtime-tests)

    add_executable(strix-dual-link-telemetry-tests tests/dual_link_telemetry_tests.cpp)
    target_link_libraries(strix-dual-link-telemetry-tests PRIVATE strix_core)
    strix_target_warnings(strix-dual-link-telemetry-tests)
    add_test(NAME strix-dual-link-telemetry-tests COMMAND strix-dual-link-telemetry-tests)

    add_executable(strix-kernel-registry-tests tests/kernel_registry_tests.cpp)
    target_link_libraries(strix-kernel-registry-tests PRIVATE strix_core)
    strix_target_warnings(strix-kernel-registry-tests)
    add_test(NAME strix-kernel-registry-tests COMMAND strix-kernel-registry-tests)

    add_executable(strix-packed-projection-tests tests/packed_projection_tests.cpp)
    target_link_libraries(strix-packed-projection-tests PRIVATE strix_core)
    strix_target_warnings(strix-packed-projection-tests)
    add_test(NAME strix-packed-projection-tests COMMAND strix-packed-projection-tests)

    add_executable(strix-iq4-xs-tests tests/iq4_xs_tests.cpp)
    target_link_libraries(strix-iq4-xs-tests PRIVATE strix_core)
    strix_target_warnings(strix-iq4-xs-tests)
    add_test(NAME strix-iq4-xs-tests COMMAND strix-iq4-xs-tests)

    add_executable(strix-iq4-nl-tests tests/iq4_nl_tests.cpp)
    target_link_libraries(strix-iq4-nl-tests PRIVATE strix_core)
    strix_target_warnings(strix-iq4-nl-tests)
    add_test(NAME strix-iq4-nl-tests COMMAND strix-iq4-nl-tests)

    add_executable(
        strix-indexed-attention-workspace-tests
        tests/indexed_attention_workspace_tests.cpp
    )
    target_link_libraries(
        strix-indexed-attention-workspace-tests
        PRIVATE strix_core
    )
    strix_target_warnings(strix-indexed-attention-workspace-tests)
    add_test(
        NAME strix-indexed-attention-workspace-tests
        COMMAND strix-indexed-attention-workspace-tests
    )

    add_executable(strix-recurrent-plan-tests tests/recurrent_plan_tests.cpp)
    target_link_libraries(strix-recurrent-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-recurrent-plan-tests)
    add_test(NAME strix-recurrent-plan-tests COMMAND strix-recurrent-plan-tests)

    add_executable(strix-moe-plan-tests tests/moe_plan_tests.cpp)
    target_link_libraries(strix-moe-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-plan-tests)
    add_test(NAME strix-moe-plan-tests COMMAND strix-moe-plan-tests)

    add_executable(strix-moe-layer-schedule-tests tests/moe_layer_schedule_tests.cpp)
    target_link_libraries(strix-moe-layer-schedule-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-layer-schedule-tests)
    add_test(NAME strix-moe-layer-schedule-tests COMMAND strix-moe-layer-schedule-tests)

    add_executable(strix-moe-layer-runtime-tests tests/moe_layer_runtime_tests.cpp)
    target_link_libraries(strix-moe-layer-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-layer-runtime-tests)
    add_test(NAME strix-moe-layer-runtime-tests COMMAND strix-moe-layer-runtime-tests)

    add_executable(strix-partition-search-tests tests/partition_search_tests.cpp)
    target_link_libraries(strix-partition-search-tests PRIVATE strix_core)
    strix_target_warnings(strix-partition-search-tests)
    add_test(NAME strix-partition-search-tests COMMAND strix-partition-search-tests)

    add_executable(strix-sparse-lookup-tests tests/sparse_lookup_tests.cpp)
    target_link_libraries(strix-sparse-lookup-tests PRIVATE strix_core)
    strix_target_warnings(strix-sparse-lookup-tests)
    add_test(NAME strix-sparse-lookup-tests COMMAND strix-sparse-lookup-tests)

    add_executable(strix-sparse-lookup-executor-tests tests/sparse_lookup_executor_tests.cpp)
    target_link_libraries(strix-sparse-lookup-executor-tests PRIVATE strix_core)
    strix_target_warnings(strix-sparse-lookup-executor-tests)
    add_test(NAME strix-sparse-lookup-executor-tests COMMAND strix-sparse-lookup-executor-tests)

    add_executable(strix-ple-lookup-tests tests/ple_lookup_tests.cpp)
    target_link_libraries(strix-ple-lookup-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-lookup-tests)
    add_test(NAME strix-ple-lookup-tests COMMAND strix-ple-lookup-tests)

    add_executable(strix-ple-direct-tests tests/ple_direct_tests.cpp)
    target_link_libraries(strix-ple-direct-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-direct-tests)
    add_test(NAME strix-ple-direct-tests COMMAND strix-ple-direct-tests)
    set_tests_properties(strix-ple-direct-tests PROPERTIES TIMEOUT 120)

    add_executable(strix-ple-rolling-tests tests/ple_rolling_tests.cpp)
    target_link_libraries(strix-ple-rolling-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-rolling-tests)
    add_test(NAME strix-ple-rolling-tests COMMAND strix-ple-rolling-tests)
    set_tests_properties(strix-ple-rolling-tests PROPERTIES TIMEOUT 120)

    add_executable(strix-ple-iq4-nl-tests tests/ple_iq4_nl_tests.cpp)
    target_link_libraries(strix-ple-iq4-nl-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-iq4-nl-tests)
    add_test(NAME strix-ple-iq4-nl-tests COMMAND strix-ple-iq4-nl-tests)

    add_executable(strix-ple-workspace-tests tests/ple_workspace_tests.cpp)
    target_link_libraries(strix-ple-workspace-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-workspace-tests)
    add_test(NAME strix-ple-workspace-tests COMMAND strix-ple-workspace-tests)

    add_executable(strix-ple-staged-gather-tests tests/ple_staged_gather_tests.cpp)
    target_link_libraries(strix-ple-staged-gather-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-staged-gather-tests)
    add_test(NAME strix-ple-staged-gather-tests COMMAND strix-ple-staged-gather-tests)

    add_executable(strix-ple-serve-policy-tests tests/ple_serve_policy_tests.cpp)
    target_link_libraries(strix-ple-serve-policy-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-serve-policy-tests)
    add_test(NAME strix-ple-serve-policy-tests COMMAND strix-ple-serve-policy-tests)

    add_executable(strix-execution-plan-tests tests/execution_plan_tests.cpp)
    target_link_libraries(strix-execution-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-plan-tests)
    add_test(NAME strix-execution-plan-tests COMMAND strix-execution-plan-tests)

    add_executable(strix-q51-grouped-down-contract-tests tests/q51_grouped_down_contract_tests.cpp)
    target_link_libraries(strix-q51-grouped-down-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-q51-grouped-down-contract-tests)
    if(NOT MSVC)
        target_compile_options(strix-q51-grouped-down-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-q51-grouped-down-contract-tests COMMAND strix-q51-grouped-down-contract-tests)

    add_executable(strix-q51-slot-parallel-contract-tests tests/q51_slot_parallel_contract_tests.cpp)
    target_link_libraries(strix-q51-slot-parallel-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-q51-slot-parallel-contract-tests)
    add_test(NAME strix-q51-slot-parallel-contract-tests COMMAND strix-q51-slot-parallel-contract-tests)

    add_executable(strix-moe-q4k-a8-runtime-tests tests/moe_q4k_a8_runtime_tests.cpp)
    target_link_libraries(strix-moe-q4k-a8-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-q4k-a8-runtime-tests)
    add_test(NAME strix-moe-q4k-a8-runtime-tests COMMAND strix-moe-q4k-a8-runtime-tests)

    add_executable(strix-execution-buffer-tests tests/execution_buffers_tests.cpp)
    target_link_libraries(strix-execution-buffer-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-buffer-tests)
    add_test(NAME strix-execution-buffer-tests COMMAND strix-execution-buffer-tests)

    add_executable(strix-execution-binding-tests tests/execution_binding_tests.cpp)
    target_link_libraries(strix-execution-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-binding-tests)
    add_test(NAME strix-execution-binding-tests COMMAND strix-execution-binding-tests)

    add_executable(strix-execution-runtime-tests tests/execution_runtime_tests.cpp)
    target_link_libraries(strix-execution-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-runtime-tests)
    add_test(NAME strix-execution-runtime-tests COMMAND strix-execution-runtime-tests)

    add_executable(strix-execution-tensor-binding-tests tests/execution_tensor_binding_tests.cpp)
    target_link_libraries(strix-execution-tensor-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-tensor-binding-tests)
    add_test(NAME strix-execution-tensor-binding-tests COMMAND strix-execution-tensor-binding-tests)

    add_executable(strix-execution-io-tests tests/execution_io_tests.cpp)
    target_link_libraries(strix-execution-io-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-io-tests)
    add_test(NAME strix-execution-io-tests COMMAND strix-execution-io-tests)

    add_executable(strix-execution-arena-tests tests/execution_arena_tests.cpp)
    target_link_libraries(strix-execution-arena-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-arena-tests)
    add_test(NAME strix-execution-arena-tests COMMAND strix-execution-arena-tests)

    add_executable(strix-qsa-score-scratch-tests tests/qsa_score_scratch_tests.cpp)
    target_link_libraries(strix-qsa-score-scratch-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-score-scratch-tests)
    add_test(NAME strix-qsa-score-scratch-tests COMMAND strix-qsa-score-scratch-tests)

    add_executable(strix-memory-plan-tests tests/memory_plan_tests.cpp)
    target_link_libraries(strix-memory-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-memory-plan-tests)
    add_test(NAME strix-memory-plan-tests COMMAND strix-memory-plan-tests)

    add_executable(strix-execution-memory-tests tests/execution_memory_tests.cpp)
    target_link_libraries(strix-execution-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-memory-tests)
    add_test(NAME strix-execution-memory-tests COMMAND strix-execution-memory-tests)

    add_executable(strix-execution-resources-tests tests/execution_resources_tests.cpp)
    target_link_libraries(strix-execution-resources-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-resources-tests)
    add_test(NAME strix-execution-resources-tests COMMAND strix-execution-resources-tests)

    add_executable(strix-runtime-owner-inventory-receipt-tests tests/runtime_owner_inventory_receipt_tests.cpp)
    target_link_libraries(strix-runtime-owner-inventory-receipt-tests PRIVATE strix_core)
    strix_target_warnings(strix-runtime-owner-inventory-receipt-tests)
    add_test(NAME strix-runtime-owner-inventory-receipt-tests COMMAND strix-runtime-owner-inventory-receipt-tests)

    add_executable(strix-linux-uma-observation-tests tests/linux_uma_observation_tests.cpp)
    target_link_libraries(strix-linux-uma-observation-tests PRIVATE strix_core)
    strix_target_warnings(strix-linux-uma-observation-tests)
    add_test(NAME strix-linux-uma-observation-tests COMMAND strix-linux-uma-observation-tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    option(STRIX_ASSET_CLI_TESTS "Require asset CLI qualification" ON)
    if(STRIX_ASSET_CLI_TESTS)
        if(Python3_Interpreter_FOUND)
            add_test(NAME strix-asset-manifest-cli-tests COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/asset_manifest_cli_test.py $<TARGET_FILE:strix-model-manifest>)
        else()
            add_test(NAME strix-asset-manifest-cli-tests COMMAND ${CMAKE_COMMAND}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/asset_cli_requires_python.cmake)
        endif()
    endif()
    add_executable(strix-asset-manifest-tests tests/asset_manifest_test.cpp src/asset_manifest.cpp src/serve/json.cpp)
    target_link_libraries(strix-asset-manifest-tests PRIVATE strix_core)
    target_compile_definitions(strix-asset-manifest-tests PRIVATE STRIX_ASSET_TEST_HOOKS=1)
    target_compile_definitions(strix-asset-manifest-tests PRIVATE STRIX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    strix_target_warnings(strix-asset-manifest-tests)
    add_test(NAME strix-asset-manifest-tests COMMAND strix-asset-manifest-tests)

    add_executable(strix-preflight-tests tests/preflight_test.cpp src/serve/json.cpp)
    target_link_libraries(strix-preflight-tests PRIVATE strix_core)
    strix_target_warnings(strix-preflight-tests)
    add_test(NAME strix-preflight-tests COMMAND strix-preflight-tests)

    add_executable(strix-build-info-tests tests/build_info_test.cpp src/serve/json.cpp)
    target_link_libraries(strix-build-info-tests PRIVATE strix_core)
    strix_target_warnings(strix-build-info-tests)
    add_test(NAME strix-build-info-tests COMMAND strix-build-info-tests)

    add_executable(strix-distributed-stage-state-tests tests/distributed_stage_state_tests.cpp)
    target_link_libraries(strix-distributed-stage-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-stage-state-tests)
    add_test(NAME strix-distributed-stage-state-tests COMMAND strix-distributed-stage-state-tests)

    add_executable(strix-distributed-greedy-control-tests tests/distributed_greedy_control_tests.cpp)
    target_link_libraries(strix-distributed-greedy-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-greedy-control-tests)
    add_test(NAME strix-distributed-greedy-control-tests COMMAND strix-distributed-greedy-control-tests)

    add_executable(strix-distributed-token-transaction-tests tests/distributed_token_transaction_tests.cpp)
    target_link_libraries(strix-distributed-token-transaction-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-token-transaction-tests)
    add_test(NAME strix-distributed-token-transaction-tests COMMAND strix-distributed-token-transaction-tests)

    add_executable(strix-distributed-placement-tests tests/distributed_placement_tests.cpp)
    target_link_libraries(strix-distributed-placement-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-placement-tests)
    add_test(NAME strix-distributed-placement-tests COMMAND strix-distributed-placement-tests)

    add_executable(strix-distributed-flow-tests tests/distributed_flow_tests.cpp)
    target_link_libraries(strix-distributed-flow-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-flow-tests)
    add_test(NAME strix-distributed-flow-tests COMMAND strix-distributed-flow-tests)

    add_executable(strix-distributed-data-plane-tests tests/distributed_data_plane_tests.cpp)
    target_link_libraries(strix-distributed-data-plane-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-data-plane-tests)
    add_test(NAME strix-distributed-data-plane-tests COMMAND strix-distributed-data-plane-tests)

    add_executable(strix-distributed-runtime-tests tests/distributed_runtime_tests.cpp)
    target_link_libraries(strix-distributed-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-runtime-tests)
    add_test(NAME strix-distributed-runtime-tests COMMAND strix-distributed-runtime-tests)

    add_executable(strix-distributed-native-backend-tests tests/distributed_native_backend_tests.cpp)
    target_link_libraries(strix-distributed-native-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-native-backend-tests)
    add_test(NAME strix-distributed-native-backend-tests COMMAND strix-distributed-native-backend-tests)

    add_executable(strix-distributed-remote-control-tests tests/distributed_remote_control_tests.cpp)
    target_link_libraries(strix-distributed-remote-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-remote-control-tests)
    add_test(NAME strix-distributed-remote-control-tests COMMAND strix-distributed-remote-control-tests)

    add_executable(strix-distributed-control-wire-tests tests/distributed_control_wire_tests.cpp)
    target_link_libraries(strix-distributed-control-wire-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-control-wire-tests)
    add_test(NAME strix-distributed-control-wire-tests COMMAND strix-distributed-control-wire-tests)

    add_executable(strix-distributed-control-server-tests tests/distributed_control_server_tests.cpp)
    target_link_libraries(strix-distributed-control-server-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-control-server-tests)
    add_test(NAME strix-distributed-control-server-tests COMMAND strix-distributed-control-server-tests)

    if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|Darwin)$")
        add_executable(strix-distributed-control-end-to-end-tests tests/distributed_control_end_to_end_tests.cpp)
        target_link_libraries(strix-distributed-control-end-to-end-tests PRIVATE strix_core)
        strix_target_warnings(strix-distributed-control-end-to-end-tests)
        add_test(NAME strix-distributed-control-end-to-end-tests COMMAND strix-distributed-control-end-to-end-tests)
    endif()

    add_executable(strix-distributed-control-receiver-tests tests/distributed_control_receiver_tests.cpp)
    target_link_libraries(strix-distributed-control-receiver-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-control-receiver-tests)
    add_test(NAME strix-distributed-control-receiver-tests COMMAND strix-distributed-control-receiver-tests)

    if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|Darwin)$")
        add_executable(strix-distributed-tcp-control-stream-tests tests/distributed_tcp_control_stream_tests.cpp)
        target_link_libraries(strix-distributed-tcp-control-stream-tests PRIVATE strix_core)
        strix_target_warnings(strix-distributed-tcp-control-stream-tests)
        add_test(NAME strix-distributed-tcp-control-stream-tests COMMAND strix-distributed-tcp-control-stream-tests)
    endif()

    add_executable(strix-distributed-session-tests tests/distributed_session_tests.cpp)
    target_link_libraries(strix-distributed-session-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-session-tests)
    add_test(NAME strix-distributed-session-tests COMMAND strix-distributed-session-tests)

    add_executable(strix-execution-tensor-access-tests tests/execution_tensor_access_tests.cpp)
    target_link_libraries(strix-execution-tensor-access-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-tensor-access-tests)
    add_test(NAME strix-execution-tensor-access-tests COMMAND strix-execution-tensor-access-tests)

    add_executable(strix-dispatch-resources-tests tests/dispatch_resources_tests.cpp)
    target_link_libraries(strix-dispatch-resources-tests PRIVATE strix_core)
    strix_target_warnings(strix-dispatch-resources-tests)
    add_test(NAME strix-dispatch-resources-tests COMMAND strix-dispatch-resources-tests)

    add_executable(strix-launch-dispatch-tests tests/launch_dispatch_tests.cpp)
    target_link_libraries(strix-launch-dispatch-tests PRIVATE strix_core)
    strix_target_warnings(strix-launch-dispatch-tests)
    add_test(NAME strix-launch-dispatch-tests COMMAND strix-launch-dispatch-tests)

    add_executable(strix-speculative-verify-launch-dispatch-tests tests/speculative_verify_launch_dispatch_tests.cpp)
    target_link_libraries(strix-speculative-verify-launch-dispatch-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-launch-dispatch-tests)
    add_test(NAME strix-speculative-verify-launch-dispatch-tests COMMAND strix-speculative-verify-launch-dispatch-tests)

    add_executable(strix-runtime-launchers-tests tests/runtime_launchers_tests.cpp)
    target_link_libraries(strix-runtime-launchers-tests PRIVATE strix_core)
    strix_target_warnings(strix-runtime-launchers-tests)
    add_test(NAME strix-runtime-launchers-tests COMMAND strix-runtime-launchers-tests)

    add_executable(strix-compiled-execution-tests tests/compiled_execution_tests.cpp)
    target_link_libraries(strix-compiled-execution-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-execution-tests)
    add_test(NAME strix-compiled-execution-tests COMMAND strix-compiled-execution-tests)

    add_executable(strix-serial-session-tests tests/serial_session_tests.cpp)
    target_link_libraries(strix-serial-session-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-session-tests)
    add_test(NAME strix-serial-session-tests COMMAND strix-serial-session-tests)

    add_executable(strix-session-executor-handoff-tests tests/session_executor_handoff_tests.cpp)
    target_link_libraries(strix-session-executor-handoff-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-executor-handoff-tests)
    add_test(NAME strix-session-executor-handoff-tests COMMAND strix-session-executor-handoff-tests)

    add_executable(strix-bitcalc-measurement-tests tests/bitcalc_measurement_tests.cpp)
    target_link_libraries(strix-bitcalc-measurement-tests PRIVATE strix_core)
    strix_target_warnings(strix-bitcalc-measurement-tests)
    add_test(NAME strix-bitcalc-measurement-tests COMMAND strix-bitcalc-measurement-tests)

    add_executable(strix-serial-request-tests tests/serial_request_tests.cpp)
    target_link_libraries(strix-serial-request-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-request-tests)
    add_test(NAME strix-serial-request-tests COMMAND strix-serial-request-tests)

    add_executable(strix-serial-prefix-lifecycle-tests tests/serial_prefix_lifecycle_tests.cpp)
    target_link_libraries(strix-serial-prefix-lifecycle-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-prefix-lifecycle-tests)
    add_test(NAME strix-serial-prefix-lifecycle-tests COMMAND strix-serial-prefix-lifecycle-tests)

    add_executable(strix-serial-prefix-cache-tests tests/serial_prefix_cache_tests.cpp)
    target_link_libraries(strix-serial-prefix-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-prefix-cache-tests)
    add_test(NAME strix-serial-prefix-cache-tests COMMAND strix-serial-prefix-cache-tests)

    add_executable(strix-serial-decode-replay-tests tests/serial_decode_replay_tests.cpp)
    target_link_libraries(strix-serial-decode-replay-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-decode-replay-tests)
    add_test(NAME strix-serial-decode-replay-tests COMMAND strix-serial-decode-replay-tests)

    add_executable(strix-serial-decode-reference-tests tests/serial_decode_reference_tests.cpp)
    target_link_libraries(strix-serial-decode-reference-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-decode-reference-tests)
    add_test(NAME strix-serial-decode-reference-tests COMMAND strix-serial-decode-reference-tests)

    add_executable(strix-session-state-layout-tests tests/session_state_layout_tests.cpp)
    target_link_libraries(strix-session-state-layout-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-state-layout-tests)
    add_test(NAME strix-session-state-layout-tests COMMAND strix-session-state-layout-tests)

    add_executable(strix-session-state-memory-tests tests/session_state_memory_tests.cpp)
    target_link_libraries(strix-session-state-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-state-memory-tests)
    add_test(NAME strix-session-state-memory-tests COMMAND strix-session-state-memory-tests)

    add_executable(strix-session-snapshot-tests tests/session_snapshot_tests.cpp)
    target_link_libraries(strix-session-snapshot-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-snapshot-tests)
    add_test(NAME strix-session-snapshot-tests COMMAND strix-session-snapshot-tests)

    add_executable(strix-session-snapshot-memory-tests tests/session_snapshot_memory_tests.cpp)
    target_link_libraries(strix-session-snapshot-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-snapshot-memory-tests)
    add_test(NAME strix-session-snapshot-memory-tests COMMAND strix-session-snapshot-memory-tests)

    add_executable(strix-session-restore-tests tests/session_restore_tests.cpp)
    target_link_libraries(strix-session-restore-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-restore-tests)
    add_test(NAME strix-session-restore-tests COMMAND strix-session-restore-tests)

    add_executable(strix-session-snapshot-transfer-tests tests/session_snapshot_transfer_tests.cpp)
    target_link_libraries(strix-session-snapshot-transfer-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-snapshot-transfer-tests)
    add_test(NAME strix-session-snapshot-transfer-tests COMMAND strix-session-snapshot-transfer-tests)

    add_executable(strix-runtime-allocator-tests tests/runtime_allocator_tests.cpp)
    target_link_libraries(strix-runtime-allocator-tests PRIVATE strix_core)
    strix_target_warnings(strix-runtime-allocator-tests)
    add_test(NAME strix-runtime-allocator-tests COMMAND strix-runtime-allocator-tests)

    add_executable(strix-speculation-tests tests/speculation_tests.cpp)
    target_link_libraries(strix-speculation-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-tests)
    add_test(NAME strix-speculation-tests COMMAND strix-speculation-tests)

    add_executable(strix-speculation-sampling-tests tests/speculation_sampling_tests.cpp)
    target_link_libraries(strix-speculation-sampling-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-tests)
    add_test(NAME strix-speculation-sampling-tests COMMAND strix-speculation-sampling-tests)

    add_executable(strix-speculation-sampling-distribution-tests tests/speculation_sampling_distribution_tests.cpp)
    target_link_libraries(strix-speculation-sampling-distribution-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-distribution-tests)
    add_test(NAME strix-speculation-sampling-distribution-tests COMMAND strix-speculation-sampling-distribution-tests)

    add_executable(strix-speculation-sampling-cycle-tests tests/speculation_sampling_cycle_tests.cpp)
    target_link_libraries(strix-speculation-sampling-cycle-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-cycle-tests)
    add_test(NAME strix-speculation-sampling-cycle-tests COMMAND strix-speculation-sampling-cycle-tests)

    add_executable(strix-speculation-sampling-executor-tests tests/speculation_sampling_executor_tests.cpp)
    target_link_libraries(strix-speculation-sampling-executor-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-executor-tests)
    add_test(NAME strix-speculation-sampling-executor-tests COMMAND strix-speculation-sampling-executor-tests)

    add_executable(strix-token-sampling-tests tests/token_sampling_tests.cpp)
    target_link_libraries(strix-token-sampling-tests PRIVATE strix_core)
    strix_target_warnings(strix-token-sampling-tests)
    add_test(NAME strix-token-sampling-tests COMMAND strix-token-sampling-tests)

    add_executable(strix-speculation-rng-tests tests/speculation_rng_tests.cpp)
    target_link_libraries(strix-speculation-rng-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-rng-tests)
    add_test(NAME strix-speculation-rng-tests COMMAND strix-speculation-rng-tests)

    add_executable(strix-speculation-controller-tests tests/speculation_controller_tests.cpp)
    target_link_libraries(strix-speculation-controller-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-controller-tests)
    add_test(NAME strix-speculation-controller-tests COMMAND strix-speculation-controller-tests)

    add_executable(strix-speculation-measurement-tests tests/speculation_measurement_tests.cpp)
    target_link_libraries(strix-speculation-measurement-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-measurement-tests)
    add_test(NAME strix-speculation-measurement-tests COMMAND strix-speculation-measurement-tests)

    add_executable(strix-speculation-cycle-tests tests/speculation_cycle_tests.cpp)
    target_link_libraries(strix-speculation-cycle-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-cycle-tests)
    add_test(NAME strix-speculation-cycle-tests COMMAND strix-speculation-cycle-tests)

    add_executable(strix-prompt-lookup-draft-tests tests/prompt_lookup_draft_tests.cpp)
    target_link_libraries(strix-prompt-lookup-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-prompt-lookup-draft-tests)
    add_test(NAME strix-prompt-lookup-draft-tests COMMAND strix-prompt-lookup-draft-tests)

    add_executable(strix-qsa-committed-pool-lease-tests tests/qsa_committed_pool_lease_tests.cpp)
    target_link_libraries(strix-qsa-committed-pool-lease-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-committed-pool-lease-tests)
    add_test(NAME strix-qsa-committed-pool-lease-tests COMMAND strix-qsa-committed-pool-lease-tests)

    add_executable(strix-mtp-step-logits-digest-tests tests/mtp_step_logits_digest_tests.cpp)
    target_link_libraries(strix-mtp-step-logits-digest-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-step-logits-digest-tests)
    add_test(NAME strix-mtp-step-logits-digest-tests COMMAND strix-mtp-step-logits-digest-tests)
    add_executable(strix-qsa-committed-pool-tests tests/qsa_committed_pool_tests.cpp)
    target_link_libraries(strix-qsa-committed-pool-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-committed-pool-tests)
    add_test(NAME strix-qsa-committed-pool-tests COMMAND strix-qsa-committed-pool-tests)

    add_executable(strix-qsa-selection-plan-tests tests/qsa_selection_plan_tests.cpp)
    target_link_libraries(strix-qsa-selection-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-selection-plan-tests)
    add_test(NAME strix-qsa-selection-plan-tests COMMAND strix-qsa-selection-plan-tests)

    add_executable(strix-compiled-mtp-draft-plan-tests tests/compiled_mtp_draft_plan_tests.cpp)
    target_link_libraries(strix-compiled-mtp-draft-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-mtp-draft-plan-tests)
    add_test(NAME strix-compiled-mtp-draft-plan-tests COMMAND strix-compiled-mtp-draft-plan-tests)

    add_executable(strix-mtp-draft-cycle-input-tests tests/mtp_draft_cycle_input_tests.cpp)
    target_link_libraries(strix-mtp-draft-cycle-input-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-cycle-input-tests)
    add_test(NAME strix-mtp-draft-cycle-input-tests COMMAND strix-mtp-draft-cycle-input-tests)

    add_executable(strix-mtp-ngram-draft-tests tests/mtp_ngram_draft_tests.cpp)
    target_link_libraries(strix-mtp-ngram-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-ngram-draft-tests)
    add_test(NAME strix-mtp-ngram-draft-tests COMMAND strix-mtp-ngram-draft-tests)

    add_executable(strix-mtp-frame-draft-tests tests/mtp_frame_draft_tests.cpp)
    target_link_libraries(strix-mtp-frame-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-frame-draft-tests)
    add_test(NAME strix-mtp-frame-draft-tests COMMAND strix-mtp-frame-draft-tests)

    add_executable(strix-mtp-context-draft-tests tests/mtp_context_draft_tests.cpp)
    target_link_libraries(strix-mtp-context-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-context-draft-tests)
    add_test(NAME strix-mtp-context-draft-tests COMMAND strix-mtp-context-draft-tests)

    add_executable(strix-mtp-draft-activation-tests tests/mtp_draft_activation_tests.cpp)
    target_link_libraries(strix-mtp-draft-activation-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-activation-tests)
    add_test(NAME strix-mtp-draft-activation-tests COMMAND strix-mtp-draft-activation-tests)

    add_executable(strix-mtp-forward-replay-tests tests/mtp_forward_replay_tests.cpp)
    target_link_libraries(strix-mtp-forward-replay-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-replay-tests)
    add_test(NAME strix-mtp-forward-replay-tests COMMAND strix-mtp-forward-replay-tests)

    add_executable(strix-hip-mtp-decode-options-tests tests/hip_mtp_decode_options_tests.cpp)
    target_link_libraries(strix-hip-mtp-decode-options-tests PRIVATE strix_core)
    strix_target_warnings(strix-hip-mtp-decode-options-tests)
    add_test(NAME strix-hip-mtp-decode-options-tests COMMAND strix-hip-mtp-decode-options-tests)

    add_executable(strix-mtp-forward-golden-file-tests tests/mtp_forward_golden_file_tests.cpp)
    target_link_libraries(strix-mtp-forward-golden-file-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-golden-file-tests)
    add_test(NAME strix-mtp-forward-golden-file-tests COMMAND strix-mtp-forward-golden-file-tests)

    add_executable(strix-mtp-target-window-policy-tests tests/mtp_target_window_policy_tests.cpp)
    target_link_libraries(strix-mtp-target-window-policy-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-target-window-policy-tests)
    add_test(NAME strix-mtp-target-window-policy-tests COMMAND strix-mtp-target-window-policy-tests)

    add_executable(strix-mtp-window-controller-tests tests/mtp_window_controller_tests.cpp)
    target_link_libraries(strix-mtp-window-controller-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-window-controller-tests)
    add_test(NAME strix-mtp-window-controller-tests COMMAND strix-mtp-window-controller-tests)
    add_executable(strix-cost-table-cache-tests tests/cost_table_cache_tests.cpp)
    target_link_libraries(strix-cost-table-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-cost-table-cache-tests)
    add_test(NAME strix-cost-table-cache-tests COMMAND strix-cost-table-cache-tests)
    add_executable(strix-mtp-depth-window-controller-tests tests/mtp_depth_window_controller_tests.cpp)
    target_link_libraries(strix-mtp-depth-window-controller-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-depth-window-controller-tests)
    add_test(NAME strix-mtp-depth-window-controller-tests COMMAND strix-mtp-depth-window-controller-tests)
    add_executable(strix-mtp-shared-execution-witness-tests tests/mtp_shared_execution_witness_tests.cpp)
    target_link_libraries(strix-mtp-shared-execution-witness-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-shared-execution-witness-tests)
    add_test(NAME strix-mtp-shared-execution-witness-tests COMMAND strix-mtp-shared-execution-witness-tests)

    add_executable(strix-mtp-sidecar-binding-tests tests/mtp_sidecar_binding_tests.cpp)
    target_link_libraries(strix-mtp-sidecar-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-sidecar-binding-tests)
    add_test(NAME strix-mtp-sidecar-binding-tests COMMAND strix-mtp-sidecar-binding-tests)

    add_executable(strix-mtp-forward-semantics-tests tests/mtp_forward_semantics_tests.cpp)
    target_link_libraries(strix-mtp-forward-semantics-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-semantics-tests)
    add_test(NAME strix-mtp-forward-semantics-tests COMMAND strix-mtp-forward-semantics-tests)

    add_executable(strix-mtp-forward-contract-tests tests/mtp_forward_contract_tests.cpp)
    target_link_libraries(strix-mtp-forward-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-contract-tests)
    add_test(NAME strix-mtp-forward-contract-tests COMMAND strix-mtp-forward-contract-tests)

    add_executable(strix-mtp-forward-runtime-tests tests/mtp_forward_runtime_tests.cpp)
    target_link_libraries(strix-mtp-forward-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-runtime-tests)
    add_test(NAME strix-mtp-forward-runtime-tests COMMAND strix-mtp-forward-runtime-tests)

    add_executable(strix-mtp-draft-readiness-tests tests/mtp_draft_readiness_tests.cpp)
    target_link_libraries(strix-mtp-draft-readiness-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-readiness-tests)
    add_test(NAME strix-mtp-draft-readiness-tests COMMAND strix-mtp-draft-readiness-tests)

    add_executable(strix-mtp-draft-qualification-tests tests/mtp_draft_qualification_tests.cpp)
    target_link_libraries(strix-mtp-draft-qualification-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-qualification-tests)
    add_test(NAME strix-mtp-draft-qualification-tests COMMAND strix-mtp-draft-qualification-tests)

    add_executable(strix-qualified-mtp-draft-backend-tests tests/qualified_mtp_draft_backend_tests.cpp)
    target_link_libraries(strix-qualified-mtp-draft-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-qualified-mtp-draft-backend-tests)
    add_test(NAME strix-qualified-mtp-draft-backend-tests COMMAND strix-qualified-mtp-draft-backend-tests)

    add_executable(strix-qualified-mtp-forward-runtime-tests tests/qualified_mtp_forward_runtime_tests.cpp)
    target_link_libraries(strix-qualified-mtp-forward-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-qualified-mtp-forward-runtime-tests)
    add_test(NAME strix-qualified-mtp-forward-runtime-tests COMMAND strix-qualified-mtp-forward-runtime-tests)

    add_executable(strix-mtp-attention-contract-tests tests/mtp_attention_contract_tests.cpp)
    target_link_libraries(strix-mtp-attention-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-attention-contract-tests)
    add_test(NAME strix-mtp-attention-contract-tests COMMAND strix-mtp-attention-contract-tests)

    add_executable(strix-mtp-memory-state-tests tests/mtp_memory_state_tests.cpp)
    target_link_libraries(strix-mtp-memory-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-memory-state-tests)
    add_test(NAME strix-mtp-memory-state-tests COMMAND strix-mtp-memory-state-tests)

    add_executable(strix-mtp-draft-state-contract-tests tests/mtp_draft_state_contract_tests.cpp)
    target_link_libraries(strix-mtp-draft-state-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-state-contract-tests)
    add_test(NAME strix-mtp-draft-state-contract-tests COMMAND strix-mtp-draft-state-contract-tests)

    add_executable(strix-mtp-draft-state-memory-tests tests/mtp_draft_state_memory_tests.cpp)
    target_link_libraries(strix-mtp-draft-state-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-state-memory-tests)
    add_test(NAME strix-mtp-draft-state-memory-tests COMMAND strix-mtp-draft-state-memory-tests)

    add_executable(strix-mtp-output-contract-tests tests/mtp_output_contract_tests.cpp)
    target_link_libraries(strix-mtp-output-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-output-contract-tests)
    add_test(NAME strix-mtp-output-contract-tests COMMAND strix-mtp-output-contract-tests)

    add_executable(strix-mtp-input-fusion-tests tests/mtp_input_fusion_tests.cpp)
    target_link_libraries(strix-mtp-input-fusion-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-input-fusion-tests)
    add_test(NAME strix-mtp-input-fusion-tests COMMAND strix-mtp-input-fusion-tests)

    add_executable(strix-mtp-draft-tensor-audit-tests tests/mtp_draft_tensor_audit_tests.cpp)
    target_link_libraries(strix-mtp-draft-tensor-audit-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-tensor-audit-tests)
    add_test(NAME strix-mtp-draft-tensor-audit-tests COMMAND strix-mtp-draft-tensor-audit-tests)

    add_executable(strix-mtp-draft-contract-tests tests/mtp_draft_contract_tests.cpp)
    target_link_libraries(strix-mtp-draft-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-contract-tests)
    add_test(NAME strix-mtp-draft-contract-tests COMMAND strix-mtp-draft-contract-tests)

    add_executable(strix-mtp-draft-binding-tests tests/mtp_draft_binding_tests.cpp)
    target_link_libraries(strix-mtp-draft-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-binding-tests)
    add_test(NAME strix-mtp-draft-binding-tests COMMAND strix-mtp-draft-binding-tests)

    add_executable(strix-draft-proposal-state-tests tests/draft_proposal_state_tests.cpp)
    target_link_libraries(strix-draft-proposal-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-draft-proposal-state-tests)
    add_test(NAME strix-draft-proposal-state-tests COMMAND strix-draft-proposal-state-tests)

    add_executable(strix-speculation-executor-tests tests/speculation_executor_tests.cpp)
    target_link_libraries(strix-speculation-executor-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-executor-tests)
    add_test(NAME strix-speculation-executor-tests COMMAND strix-speculation-executor-tests)

    add_executable(strix-compiled-speculation-backend-tests tests/compiled_speculation_backend_tests.cpp)
    target_link_libraries(strix-compiled-speculation-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-speculation-backend-tests)
    add_test(NAME strix-compiled-speculation-backend-tests COMMAND strix-compiled-speculation-backend-tests)

    add_executable(strix-compiled-sampled-speculation-backend-tests tests/compiled_sampled_speculation_backend_tests.cpp)
    target_link_libraries(strix-compiled-sampled-speculation-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-sampled-speculation-backend-tests)
    add_test(NAME strix-compiled-sampled-speculation-backend-tests COMMAND strix-compiled-sampled-speculation-backend-tests)

    add_executable(strix-speculative-verify-plan-tests tests/speculative_verify_plan_tests.cpp)
    target_link_libraries(strix-speculative-verify-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-plan-tests)
    add_test(NAME strix-speculative-verify-plan-tests COMMAND strix-speculative-verify-plan-tests)

    add_executable(strix-speculative-verify-binding-tests tests/speculative_verify_binding_tests.cpp)
    target_link_libraries(strix-speculative-verify-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-binding-tests)
    add_test(NAME strix-speculative-verify-binding-tests COMMAND strix-speculative-verify-binding-tests)

    add_executable(strix-compiled-speculative-verify-tests tests/compiled_speculative_verify_tests.cpp)
    target_link_libraries(strix-compiled-speculative-verify-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-speculative-verify-tests)
    add_test(NAME strix-compiled-speculative-verify-tests COMMAND strix-compiled-speculative-verify-tests)
    add_test(NAME strix-compiled-speculative-verify-compact-images-off-tests COMMAND strix-compiled-speculative-verify-tests)
    set_tests_properties(strix-compiled-speculative-verify-compact-images-off-tests PROPERTIES ENVIRONMENT "STRIX_QPF2K_COMPACT_IMAGES=0")

    add_executable(strix-speculative-verify-runtime-tests tests/speculative_verify_runtime_tests.cpp)
    target_link_libraries(strix-speculative-verify-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-runtime-tests)
    add_test(NAME strix-speculative-verify-runtime-tests COMMAND strix-speculative-verify-runtime-tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-qwen-yarn-python-check
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/oracles/qwen_yarn_table_check.py
                --emitter $<TARGET_FILE:strix-qwen-yarn-table-tests>)
        add_test(NAME strix-session-gate-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/session_gate_tests.py)
        add_test(NAME strix-ds41-pair-setup-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/ds41_pair_setup_tests.py)
        add_test(NAME strix-qwen-prefill-public-controls-gate-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/qwen_prefill_public_controls_gate_tests.py)
        add_test(NAME strix-ctest-evidence-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest_evidence_tests.py)
        add_test(NAME strix-queue-trace-evidence-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/queue_trace_evidence_tests.py)
        add_test(NAME strix-runtime-inventory-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/runtime_inventory_tests.py)
    endif()
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-dual-transport-runner-tests
            COMMAND ${Python3_EXECUTABLE} -B -W error::ResourceWarning ${CMAKE_SOURCE_DIR}/lab/transport/test_dual_tbstream.py)
        add_test(NAME strix-private-kernel-preparation-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/lab/direct_gpu_stream/kernel/test_build_private_kernel.py)
        add_test(NAME strix-native-numerical-campaign-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/lab/native_campaign/numerical/test_campaign.py)
        add_test(NAME strix-native-raw-import-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/lab/native_campaign/numerical/test_raw_import.py)
        add_test(
            NAME strix-python-oracle-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/oracles/test_qwen4_exp_oracle.py
        )
        add_test(
            NAME strix-ple-qualification-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/lab/sparse_lookup/test_qualify_serial_ple.py
        )
        add_test(
            NAME strix-ple-policy-sweep-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/lab/sparse_lookup/test_sweep_serial_ple.py
        )
    endif()
endif()


# One list of AMD GPU targets for every HIP target (a successor arch is added here, e.g. "gfx1151;gfx1160"; the
# runtime gates still decide which kernel sets a qualified arch may run).
set(STRIX_HIP_ARCHITECTURES "gfx1151" CACHE STRING "AMD GPU targets (HIP_ARCHITECTURES) of every HIP target")

if(STRIX_ENABLE_HIP_RUNTIME)
    enable_language(HIP)

    # CMake's HIP language drives device compilation directly. ROCm 7.x
    # development images do not necessarily ship the legacy hipConfig.cmake /
    # hip::host package target, so locate the runtime header/library from the
    # toolkit instead of requiring that package surface.
    find_path(
        STRIX_HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        HINTS "$ENV{ROCM_PATH}" /opt/rocm
        PATH_SUFFIXES include
        REQUIRED
    )
    find_library(
        STRIX_HIP_RUNTIME_LIBRARY
        NAMES amdhip64
        HINTS "$ENV{ROCM_PATH}" /opt/rocm
        PATH_SUFFIXES lib lib64
        REQUIRED
    )
    # The HIP compiler treats the toolkit include directory as implicit and searches /usr/include first, so a
    # distribution HIP development package there (another HIP version) silently wins over the toolkit headers
    # the runtime library belongs to. Put the toolkit headers first for every HIP translation unit.
    add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:SHELL:-isystem ${STRIX_HIP_INCLUDE_DIR}>")

    if(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
        # Only these isolated arithmetic TUs receive the strict experimental flags.
        add_library(strix_q4k_a8_hip_probe STATIC src/hip_moe_q4k_a8_probe.hip src/hip_moe_q4k_a8_known_rows.hip)
        target_link_libraries(strix_q4k_a8_hip_probe PUBLIC strix_q4k_a8_contract "${STRIX_HIP_RUNTIME_LIBRARY}")
        foreach(target strix_q4k_a8_hip_probe)
            target_include_directories(${target} PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(${target} PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(${target})
        endforeach()
        if(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE)
            add_library(strix_q4k_a8_dot4_compile_probe OBJECT lab/q4k_a8/dot4_compile_probe.hip)
            target_include_directories(strix_q4k_a8_dot4_compile_probe PRIVATE include "${STRIX_HIP_INCLUDE_DIR}")
            set_target_properties(strix_q4k_a8_dot4_compile_probe PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(strix_q4k_a8_dot4_compile_probe PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix_q4k_a8_dot4_compile_probe)
        endif()
        if(BUILD_TESTING AND STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE)
            add_executable(strix-hip-moe-a8-down-tests tests/hip_moe_a8_down_tests.hip)
            target_link_libraries(strix-hip-moe-a8-down-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core strix_hip_runtime)
            set_target_properties(strix-hip-moe-a8-down-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(strix-hip-moe-a8-down-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-a8-down-tests)
            add_test(NAME strix-hip-moe-a8-down-tests COMMAND strix-hip-moe-a8-down-tests)
            add_executable(strix-hip-moe-wmma-prefill-tests tests/hip_moe_wmma_prefill_tests.hip)
            target_link_libraries(strix-hip-moe-wmma-prefill-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core)
            set_target_properties(strix-hip-moe-wmma-prefill-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(strix-hip-moe-wmma-prefill-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-wmma-prefill-tests)
            add_test(NAME strix-hip-moe-wmma-prefill-tests COMMAND strix-hip-moe-wmma-prefill-tests)
            add_test(NAME strix-hip-moe-route-bf16-tests COMMAND strix-hip-moe-wmma-prefill-tests route-bf16)
            add_test(NAME strix-hip-moe-prefill-pool16k-tests COMMAND strix-hip-moe-wmma-prefill-tests pool16k)
            add_test(NAME strix-hip-moe-prefill-schedule-tests COMMAND strix-hip-moe-wmma-prefill-tests schedule)
            add_test(NAME strix-hip-moe-prefill-gsq-tests COMMAND strix-hip-moe-wmma-prefill-tests gsq)
            add_executable(strix-hip-q4k-a8-probe-tests tests/hip_moe_q4k_a8_probe_tests.hip $<TARGET_OBJECTS:strix_q4k_a8_dot4_compile_probe>)
            target_link_libraries(strix-hip-q4k-a8-probe-tests PRIVATE strix_q4k_a8_hip_probe strix_core)
            set_target_properties(strix-hip-q4k-a8-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(strix-hip-q4k-a8-probe-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q4k-a8-probe-tests)
            add_test(NAME strix-hip-q4k-a8-probe-tests COMMAND strix-hip-q4k-a8-probe-tests)
            add_executable(strix-hip-q5k-a8-probe-tests tests/hip_moe_q5k_a8_probe_tests.hip)
            target_link_libraries(strix-hip-q5k-a8-probe-tests PRIVATE strix_q4k_a8_hip_probe strix_core)
            set_target_properties(strix-hip-q5k-a8-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(strix-hip-q5k-a8-probe-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q5k-a8-probe-tests)
            add_test(NAME strix-hip-q5k-a8-probe-tests COMMAND strix-hip-q5k-a8-probe-tests)
            add_executable(strix-hip-q20-a8-probe-tests tests/hip_moe_q20_a8_probe_tests.hip)
            target_link_libraries(strix-hip-q20-a8-probe-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core strix_hip_runtime)
            set_target_properties(strix-hip-q20-a8-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(strix-hip-q20-a8-probe-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q20-a8-probe-tests)
            add_test(NAME strix-hip-q20-a8-probe-tests COMMAND strix-hip-q20-a8-probe-tests)
        endif()
    endif()

    add_library(strix_hip_runtime STATIC
        src/hip_host_upload_backend.cpp
        src/hip_q6p_pack.cpp
        src/hip_dense_trellis_pack.cpp
        src/hip_runtime_allocator.cpp
        src/hip_sampled_target_logits_reader.cpp
        src/hip_session_snapshot_transfer.cpp
        src/hip_model_residency.cpp
        src/hip_launcher_catalog.cpp
        src/hip_runtime_launchers.hip
        src/hip_moe_route_launcher.hip
        src/hip_moe_experts_launcher.hip
        src/hip_ple_launcher.hip
        src/hip_mtp_resources.hip
        src/hip_mtp_runtime.hip
        src/hip_mtp_draft_head.hip
        src/hip_mtp_fast_draft.hip
        src/hip_mtp_target_window.hip
        src/hip_mtp_target_checkpoint.hip
        src/hip_mtp_window_verifier.hip
        src/hip_token_sampling.hip
        src/hip_token_sampling_device.hip
        src/hip_mtp_target_hidden.hip
        src/hip_mtp_greedy_runtime.hip
        src/hip_mtp_asset_loader.hip
        src/hip_mtp_input_fusion.hip
        src/hip_mtp_forward_replay.hip
        src/hip_embedding_launcher.hip
        src/hip_gdn_launcher.hip
        src/hip_known_prefix_window.hip
        src/hip_known_prompt_backend.cpp
        src/hip_prompt_lookup_runtime.hip
        src/hip_serving_greedy_backend.hip
        src/hip_serving_cohort.hip
        src/hip_serving_prefix_cache.hip
        src/hip_event_queue_trace.hip
        src/hip_distributed_stage.hip
        src/hip_distributed_transaction.hip
        src/hip_distributed_sequence.hip
        src/hip_distributed_stage_resources.hip
        src/hip_distributed_greedy.hip
        src/hip_hyper_mix_launcher.hip
        src/hip_qwen_rotary_table.hip
        src/hip_indexed_attention_launcher.hip
        src/hip_qsa_selection_diagnostic.hip
        src/hip_serial_graph.hip
        src/hip_direct_transport_gpu_export.cpp
        src/hip_direct_gpu_pool_allocator.cpp
        src/hip_direct_gpu_stream_gpu.cpp
        src/hip_integer_projection.hip
        src/hip_integer_projection_runtime.hip
        src/hip_lm_head_launcher.hip
        src/hip_generation_output.hip
        src/hip_moe_overlap_dispatcher.hip
        src/hip_shared_expert_launcher.hip
        src/hip_dense_trellis.hip
    )
    target_include_directories(
        strix_hip_runtime
        PUBLIC
        include
    )
    target_include_directories(strix_hip_runtime SYSTEM PUBLIC "${STRIX_HIP_INCLUDE_DIR}")
    if(STRIX_ENABLE_EXPERIMENTAL_DENSE_TRELLIS)
        target_compile_definitions(strix_hip_runtime PUBLIC STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS=1)
    endif()
    target_compile_features(strix_hip_runtime PUBLIC cxx_std_20)
    # Host-side redirects only: SDK declarations are included before macros;
    # kernels are never counted as submits by this observer.
    target_compile_options(strix_hip_runtime PRIVATE
        "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
    target_compile_definitions(
        strix_hip_runtime
        PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__=1>
    )
    set_target_properties(
        strix_hip_runtime
        PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
        HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
    )
    target_link_libraries(
        strix_hip_runtime
        PUBLIC
        strix_core
        "${STRIX_HIP_RUNTIME_LIBRARY}"
    )
    if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
        target_link_libraries(strix_hip_runtime PUBLIC strix_q4k_a8_hip_probe)
        target_compile_definitions(strix_hip_runtime PUBLIC STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    endif()
    set_source_files_properties(src/hip_qsa_selection_diagnostic.hip
        PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64;-fno-fast-math")
    set_source_files_properties(src/hip_dense_trellis.hip
        PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fdenormal-fp-math=ieee")
    set_source_files_properties(src/hip_mtp_input_fusion.hip
        PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee")
    strix_target_warnings(strix_hip_runtime)
    if(BUILD_TESTING)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GDN_RECURRENT_LAYOUT_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_SHARED_EXPERT_ROWS_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_QSA_GQA_REUSE_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_QWEN_ROTARY_DIAGNOSTIC=1)
        add_executable(strix-hip-qwen-yarn-rotary-tests tests/hip_qwen_yarn_rotary_tests.hip)
        target_link_libraries(strix-hip-qwen-yarn-rotary-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-qwen-yarn-rotary-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        strix_target_warnings(strix-hip-qwen-yarn-rotary-tests)
        add_test(NAME strix-hip-qwen-yarn-rotary-tests COMMAND strix-hip-qwen-yarn-rotary-tests)
        add_test(NAME strix-qwen-rotary-host-admission-tests COMMAND strix-hip-qwen-yarn-rotary-tests --host-admission-only)
        add_test(NAME strix-qwen-rotary-watchdog-tests COMMAND ${CMAKE_COMMAND}
            -DFIXTURE=$<TARGET_FILE:strix-hip-qwen-yarn-rotary-tests>
            -P ${CMAKE_SOURCE_DIR}/tests/qwen_rotary_watchdog_test.cmake)
        set_tests_properties(strix-hip-qwen-yarn-rotary-tests PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_QPERSIST_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC=1)
        foreach(mtp_fixture input-fusion forward-replay target-checkpoint draft-head fast-draft draft-batch)
            string(REPLACE "-" "_" mtp_fixture_file "${mtp_fixture}")
            add_executable(strix-hip-mtp-${mtp_fixture}-tests tests/hip_mtp_${mtp_fixture_file}_tests.hip)
            target_link_libraries(strix-hip-mtp-${mtp_fixture}-tests PRIVATE strix_hip_runtime)
            set_target_properties(strix-hip-mtp-${mtp_fixture}-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(strix-hip-mtp-${mtp_fixture}-tests PRIVATE
                -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off)
            strix_target_warnings(strix-hip-mtp-${mtp_fixture}-tests)
            add_test(NAME strix-hip-mtp-${mtp_fixture}-tests COMMAND strix-hip-mtp-${mtp_fixture}-tests)
            set_tests_properties(strix-hip-mtp-${mtp_fixture}-tests PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
        # Callable fixture requires actual independently supplied component goldens.
        # Building this source library does not execute or qualify the components.
        add_library(strix_mtp_projection_component_fixtures STATIC tests/hip_mtp_projection_components.hip tests/hip_mtp_shared_expert_components.hip tests/hip_mtp_target_window_components.hip tests/hip_mtp_greedy_cycle_components.hip)
        target_sources(strix_mtp_projection_component_fixtures PRIVATE
            tests/hip_mtp_lm_head_components.hip
            tests/hip_mtp_target_checkpoint_components.hip
            tests/hip_mtp_qualification_reference.hip)
        target_link_libraries(strix_mtp_projection_component_fixtures PRIVATE strix_hip_runtime)
        set_target_properties(strix_mtp_projection_component_fixtures PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        strix_target_warnings(strix_mtp_projection_component_fixtures)
        add_library(strix_qsa_selection_diagnostic INTERFACE)
        target_link_libraries(strix_qsa_selection_diagnostic INTERFACE strix_hip_runtime)
        add_executable(strix-hip-qsa-selection-diagnostic-tests tests/hip_qsa_selection_diagnostic_tests.hip)
        target_link_libraries(strix-hip-qsa-selection-diagnostic-tests PRIVATE strix_qsa_selection_diagnostic)
        set_target_properties(strix-hip-qsa-selection-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        strix_target_warnings(strix-hip-qsa-selection-diagnostic-tests)
        add_test(NAME strix-hip-qsa-selection-diagnostic-tests COMMAND strix-hip-qsa-selection-diagnostic-tests)
    endif()
    set_source_files_properties(src/hip_integer_projection.hip src/hip_integer_projection_runtime.hip
        PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64;-ffp-contract=off")

    add_executable(
        strix-hip-runtime-allocator
        apps/hip_runtime_allocator_main.cpp
    )

    add_executable(
        strix-hip-model-residency
        apps/hip_model_residency_main.cpp
    )

    add_executable(
        strix-hip-decode-token
        apps/hip_decode_token_main.cpp
    )
    # This embeds DECLARED configuration only. Heterogeneous per-source HIP
    # argv is not represented completely, so live flags SHA stays unknown.
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" STRIX_DECLARED_CMAKE_SOURCE_SHA)
    string(HEX "cmake=${STRIX_DECLARED_CMAKE_SOURCE_SHA};cxx=${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION}/${CMAKE_CXX_COMPILER};hip=${CMAKE_HIP_COMPILER_ID}/${CMAKE_HIP_COMPILER_VERSION}/${CMAKE_HIP_COMPILER};build=${CMAKE_BUILD_TYPE};cxxflags=${CMAKE_CXX_FLAGS};hipflags=${CMAKE_HIP_FLAGS};debug=${CMAKE_HIP_FLAGS_DEBUG};release=${CMAKE_HIP_FLAGS_RELEASE};reldeb=${CMAKE_HIP_FLAGS_RELWITHDEBINFO};minsize=${CMAKE_HIP_FLAGS_MINSIZEREL};per-source=unresolved" STRIX_DECLARED_BUILD_CONTEXT_HEX)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(apps/offline_tuning_build_context.hpp.in
        "${CMAKE_CURRENT_BINARY_DIR}/generated/offline_tuning_build_context.hpp" @ONLY)
    target_include_directories(strix-hip-decode-token PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(
        strix-hip-decode-token
        PRIVATE strix_hip_runtime strix_serve strix_assets
    )
    strix_target_warnings(
        strix-hip-decode-token
    )
    target_compile_options(strix-hip-decode-token PRIVATE
        "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
    if(STRIX_CHECK_KERNEL_SCRATCH)
        strix_check_kernel_scratch(strix-hip-decode-token)
    endif()
    if(BUILD_TESTING)
        # Same actual model/PLE/asset/replay setup as the decoder. External
        # references and the serial-oracle prefix diagnostic have separate
        # admission; neither establishes model fidelity or performance alone.
        add_executable(strix-hip-mtp-model-qualification
            apps/hip_decode_token_main.cpp tests/hip_mtp_model_qualification_main.cpp)
        target_compile_definitions(strix-hip-mtp-model-qualification PRIVATE
            STRIX_DECODE_ENTRY=strix_hip_decode_token_entry STRIX_MTP_MODEL_QUALIFICATION_RUNNER=1)
        target_include_directories(strix-hip-mtp-model-qualification PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
        target_link_libraries(strix-hip-mtp-model-qualification PRIVATE
            strix_hip_runtime strix_mtp_projection_component_fixtures strix_serve strix_assets)
        # Large state snapshots are diagnostic-only. Prefer the system SHA-256
        # provider when available; the portable digest remains the fallback.
        find_package(OpenSSL QUIET COMPONENTS Crypto)
        if(TARGET OpenSSL::Crypto)
            target_link_libraries(strix-hip-mtp-model-qualification PRIVATE OpenSSL::Crypto)
            target_compile_definitions(strix-hip-mtp-model-qualification PRIVATE STRIX_PREFIX_GATE_OPENSSL_SHA256=1)
        endif()
        target_compile_options(strix-hip-mtp-model-qualification PRIVATE
            "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
        strix_target_warnings(strix-hip-mtp-model-qualification)
        add_executable(strix-hip-mtp-qualification-reference-tests tests/hip_mtp_qualification_reference_tests.hip)
        target_link_libraries(strix-hip-mtp-qualification-reference-tests PRIVATE
            strix_hip_runtime strix_mtp_projection_component_fixtures strix_serve strix_assets)
        set_target_properties(strix-hip-mtp-qualification-reference-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        strix_target_warnings(strix-hip-mtp-qualification-reference-tests)
        add_test(NAME strix-hip-mtp-qualification-reference-tests COMMAND strix-hip-mtp-qualification-reference-tests)
    endif()
    if(BUILD_TESTING)
        add_test(NAME strix-virtual-q8-gdn-output-cli-tests
            COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:strix-hip-decode-token>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_virtual_q8_gdn_output_cli.cmake)
        add_test(NAME strix-virtual-q8-cli-duplicate-tests
            COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:strix-hip-decode-token>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_virtual_q8_cli.cmake)
    endif()
    target_link_libraries(
        strix-hip-model-residency
        PRIVATE strix_hip_runtime
    )
    strix_target_warnings(
        strix-hip-model-residency
    )
    target_link_libraries(
        strix-hip-runtime-allocator
        PRIVATE strix_hip_runtime
    )
    strix_target_warnings(strix-hip-runtime-allocator)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(strix-direct-gpu-commission
            lab/direct_gpu_stream/commission_main.hip
            lab/direct_gpu_stream/bounded_tcp_control.cpp)
        set_target_properties(strix-direct-gpu-commission PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-direct-gpu-commission PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-direct-gpu-commission)
    endif()

    if(BUILD_TESTING)
        add_executable(
            strix-hip-runtime-launchers-tests
            tests/hip_runtime_launchers_tests.hip
        )
        set_target_properties(
            strix-hip-runtime-launchers-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-runtime-launchers-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-runtime-launchers-tests
        )
        add_test(
            NAME strix-hip-runtime-launchers-tests
            COMMAND strix-hip-runtime-launchers-tests
        )

        add_executable(
            strix-hip-moe-route-launcher-tests
            tests/hip_moe_route_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-moe-route-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-moe-route-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-moe-route-launcher-tests
        )
        add_test(
            NAME strix-hip-moe-route-launcher-tests
            COMMAND strix-hip-moe-route-launcher-tests
        )

        add_executable(
            strix-hip-embedding-launcher-tests
            tests/hip_embedding_launcher_tests.hip
        )
        add_executable(strix-hip-graph-inputs-tests tests/hip_graph_inputs_tests.hip)
        add_executable(strix-hip-integer-projection-runtime-tests tests/hip_integer_projection_runtime_tests.hip)
        add_executable(strix-hip-serial-graph-tests tests/hip_serial_graph_tests.hip)
        set_target_properties(strix-hip-serial-graph-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-serial-graph-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-serial-graph-tests)
        add_test(NAME strix-hip-serial-graph-tests COMMAND strix-hip-serial-graph-tests)
        set_tests_properties(strix-hip-serial-graph-tests PROPERTIES SKIP_RETURN_CODE 77)
        set_target_properties(strix-hip-integer-projection-runtime-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-integer-projection-runtime-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-integer-projection-runtime-tests PRIVATE -ffp-contract=off)
        strix_target_warnings(strix-hip-integer-projection-runtime-tests)
        add_test(NAME strix-hip-integer-projection-runtime-tests COMMAND strix-hip-integer-projection-runtime-tests)
        set_tests_properties(strix-hip-integer-projection-runtime-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-direct-transport-gpu-export-tests tests/hip_direct_transport_gpu_export_tests.cpp)
        add_executable(strix-hip-direct-gpu-pool-allocator-tests tests/hip_direct_gpu_pool_allocator_tests.cpp)
        add_executable(strix-hip-direct-gpu-stream-gpu-tests tests/hip_direct_gpu_stream_gpu_tests.cpp)
        target_link_libraries(strix-hip-direct-gpu-stream-gpu-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-direct-gpu-stream-gpu-tests)
        add_test(NAME strix-hip-direct-gpu-stream-gpu-tests COMMAND strix-hip-direct-gpu-stream-gpu-tests)
        target_link_libraries(strix-hip-direct-gpu-pool-allocator-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-direct-gpu-pool-allocator-tests)
        add_test(NAME strix-hip-direct-gpu-pool-allocator-tests COMMAND strix-hip-direct-gpu-pool-allocator-tests)
        add_executable(strix-hip-launch-context-tests tests/hip_launch_context_tests.cpp)
        target_link_libraries(strix-hip-launch-context-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-launch-context-tests)
        add_test(NAME strix-hip-launch-context-tests COMMAND strix-hip-launch-context-tests)
        target_link_libraries(strix-hip-direct-transport-gpu-export-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-direct-transport-gpu-export-tests)
        add_test(NAME strix-hip-direct-transport-gpu-export-tests COMMAND strix-hip-direct-transport-gpu-export-tests)
        add_executable(strix-hip-graph-attention-state-tests tests/hip_graph_attention_state_tests.hip)
        set_target_properties(strix-hip-graph-attention-state-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-graph-attention-state-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-graph-attention-state-tests)
        add_test(NAME strix-hip-graph-attention-state-tests COMMAND strix-hip-graph-attention-state-tests)
        set_target_properties(strix-hip-graph-inputs-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-graph-inputs-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-graph-inputs-tests)
        add_test(NAME strix-hip-graph-inputs-tests COMMAND strix-hip-graph-inputs-tests)
        set_target_properties(
            strix-hip-embedding-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-embedding-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-embedding-launcher-tests
        )
        add_test(
            NAME strix-hip-embedding-launcher-tests
            COMMAND strix-hip-embedding-launcher-tests
        )

        add_test(NAME strix-lane-chain-cli-tests COMMAND ${CMAKE_COMMAND} -DAPP=$<TARGET_FILE:strix-hip-decode-token> -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_lane_chain_cli.cmake)
        add_test(NAME strix-hip-lane-chain-hyper-integration-tests COMMAND strix-hip-hyper-mix-launcher-tests lane-chain)
        add_test(NAME strix-hip-hc-next-tile-integration-tests COMMAND strix-hip-hyper-mix-launcher-tests next-tile)
        add_executable(strix-hip-hc-projection-next-tile-tests tests/hip_hc_projection_next_tile_tests.hip)
        set_target_properties(strix-hip-hc-projection-next-tile-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-hc-projection-next-tile-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-hc-projection-next-tile-tests)
        add_test(NAME strix-hip-hc-projection-next-tile-tests COMMAND strix-hip-hc-projection-next-tile-tests)
        add_executable(strix-hip-lane-chain-hyper-output-tests tests/hip_lane_chain_hyper_output_tests.hip)
        set_target_properties(strix-hip-lane-chain-hyper-output-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-lane-chain-hyper-output-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-lane-chain-hyper-output-tests)
        add_test(NAME strix-hip-lane-chain-hyper-output-tests COMMAND strix-hip-lane-chain-hyper-output-tests)
        add_executable(strix-hip-hc-wide-rows-tests tests/hip_hc_wide_rows_tests.hip)
        set_target_properties(strix-hip-hc-wide-rows-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-hc-wide-rows-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-hc-wide-rows-tests)
        add_test(NAME strix-hip-hc-wide-rows-tests COMMAND strix-hip-hc-wide-rows-tests)
        add_test(NAME strix-hip-hc-out16-tests COMMAND strix-hip-hc-wide-rows-tests)
        set_tests_properties(strix-hip-hc-out16-tests PROPERTIES ENVIRONMENT "STRIX_PREFILL_EXACT_FP16=0;STRIX_PREFILL_HC_OUTPUT_WMMA=1;STRIX_HCW_OUTPUT=1;STRIX_HC_OUT16_FIXTURES_ONLY=1;STRIX_QPF2K_HC=1;STRIX_QPF2K_HC_LOWBAND=3")
        add_executable(strix-hip-parallel-hyper-projection-tests tests/hip_parallel_hyper_projection_tests.hip)
        set_target_properties(strix-hip-parallel-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-parallel-hyper-projection-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-parallel-hyper-projection-tests)
        add_test(NAME strix-hip-parallel-hyper-projection-tests COMMAND strix-hip-parallel-hyper-projection-tests)
        add_executable(strix-hip-splitk-hyper-integration-tests tests/hip_splitk_hyper_integration_tests.cpp)
        target_link_libraries(strix-hip-splitk-hyper-integration-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-splitk-hyper-integration-tests)
        add_test(NAME strix-hip-splitk-hyper-integration-tests COMMAND strix-hip-splitk-hyper-integration-tests)
        add_executable(strix-hip-hc-deferred-norm-tests tests/hip_hc_deferred_norm_tests.hip)
        target_link_libraries(strix-hip-hc-deferred-norm-tests PRIVATE strix_hip_runtime strix_q4k_a8_contract)
        set_target_properties(strix-hip-hc-deferred-norm-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(strix-hip-hc-deferred-norm-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-hc-deferred-norm-tests)
        add_test(NAME strix-hip-hc-deferred-norm-tests COMMAND strix-hip-hc-deferred-norm-tests)

        add_executable(strix-hip-lane-chain-hyper-projection-tests tests/hip_lane_chain_hyper_projection_tests.hip)
        set_target_properties(strix-hip-lane-chain-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-lane-chain-hyper-projection-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-lane-chain-hyper-projection-tests)
        add_test(NAME strix-hip-lane-chain-hyper-projection-tests COMMAND strix-hip-lane-chain-hyper-projection-tests)
        add_library(strix_hc_repacked_diagnostic STATIC src/hip_hc_repacked_diagnostic.hip)
        target_include_directories(strix_hc_repacked_diagnostic PUBLIC include)
        set_target_properties(strix_hc_repacked_diagnostic PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix_hc_repacked_diagnostic PUBLIC strix_core "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix_hc_repacked_diagnostic)
        add_executable(strix-hip-repacked-hyper-projection-tests tests/hip_repacked_hyper_projection_tests.hip)
        set_target_properties(strix-hip-repacked-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-repacked-hyper-projection-tests PRIVATE strix_hip_runtime strix_hc_repacked_down_pack strix_hc_repacked_diagnostic)
        strix_target_warnings(strix-hip-repacked-hyper-projection-tests)
        # Always build the diagnostic binary; only an externally admitted subset
        # registers its research test. CTest receives the cache value explicitly,
        # independent of the parent environment stripped by native workers.
        if(NOT "${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}" STREQUAL "")
            add_test(NAME strix-hip-repacked-hyper-projection-tests COMMAND strix-hip-repacked-hyper-projection-tests)
            set_tests_properties(strix-hip-repacked-hyper-projection-tests PROPERTIES
                ENVIRONMENT "STRIX_HC_REPACKED_ADMITTED_SCHEDULES=${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}"
                LABELS "research;externally-admitted-hc")
            message(STATUS "HC repacked research CTest registered for externally admitted schedules: ${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}")
        else()
            message(STATUS "HC repacked research CTest NOT registered: no schedules externally admitted; binary and portable pack contracts remain built")
        endif()
        add_executable(
            strix-hip-hyper-mix-launcher-tests
            tests/hip_hyper_mix_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-hyper-mix-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-hyper-mix-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-hyper-mix-launcher-tests
        )
        add_test(
            NAME strix-hip-hyper-mix-launcher-tests
            COMMAND strix-hip-hyper-mix-launcher-tests
        )
        add_test(NAME strix-hip-hyper-row-prefetch-tests
                 COMMAND strix-hip-hyper-mix-launcher-tests prefetch)
        add_test(NAME strix-hip-hyper-output-prefetch-tests
                 COMMAND strix-hip-hyper-mix-launcher-tests output-prefetch)

        if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
            if(UNIX)
                add_executable(strix-http-batch-session-tests tests/hip_http_batch_session_tests.cpp)
                target_compile_definitions(strix-http-batch-session-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
                target_link_libraries(strix-http-batch-session-tests PRIVATE strix_hip_runtime strix_serve)
                strix_target_warnings(strix-http-batch-session-tests)
                add_test(NAME strix-http-batch-session-tests COMMAND strix-http-batch-session-tests)
                set_tests_properties(strix-http-batch-session-tests PROPERTIES LABELS "cpu" RUN_SERIAL TRUE TIMEOUT 30)
            endif()
            add_executable(strix-qwen-yarn-host-compile-tests tests/qwen_yarn_host_compile_tests.cpp)
            target_compile_definitions(strix-qwen-yarn-host-compile-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
            target_link_libraries(strix-qwen-yarn-host-compile-tests PRIVATE strix_hip_runtime strix_serve)
            strix_target_warnings(strix-qwen-yarn-host-compile-tests)
            add_test(NAME strix-qwen-yarn-host-compile-tests COMMAND strix-qwen-yarn-host-compile-tests)
            set_tests_properties(strix-qwen-yarn-host-compile-tests PROPERTIES LABELS "cpu" RUN_SERIAL TRUE TIMEOUT 180)
        endif()

        add_executable(
            strix-hip-launcher-catalog-tests
            tests/hip_launcher_catalog_tests.cpp
        )
        target_link_libraries(
            strix-hip-launcher-catalog-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-launcher-catalog-tests
        )
        add_test(
            NAME strix-hip-launcher-catalog-tests
            COMMAND strix-hip-launcher-catalog-tests
        )

        add_executable(
            strix-hip-lm-head-launcher-tests
            tests/hip_lm_head_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-lm-head-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-lm-head-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-lm-head-launcher-tests
        )
        add_test(
            NAME strix-hip-lm-head-launcher-tests
            COMMAND strix-hip-lm-head-launcher-tests
        )

        add_executable(
            strix-hip-shared-expert-launcher-tests
            tests/hip_shared_expert_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-shared-expert-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-shared-expert-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-shared-expert-launcher-tests
        )
        add_test(
            NAME strix-hip-shared-expert-launcher-tests
            COMMAND strix-hip-shared-expert-launcher-tests
        )

        add_executable(strix-hip-shared-known-window-rows-tests tests/hip_shared_known_window_rows_tests.hip)
        set_target_properties(strix-hip-shared-known-window-rows-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-shared-known-window-rows-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-shared-known-window-rows-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-shared-known-window-rows-tests)
        foreach(rows 4 8)
            foreach(prefetch 0 1)
                add_test(NAME strix-hip-shared-known-window-rows-${rows}-prefetch-${prefetch}
                    COMMAND strix-hip-shared-known-window-rows-tests ${rows} ${prefetch})
            endforeach()
        endforeach()

        add_test(NAME strix-hip-shared-pre-fixtures COMMAND strix-hip-shared-known-window-rows-tests)
        set_tests_properties(strix-hip-shared-pre-fixtures PROPERTIES ENVIRONMENT "STRIX_SHARED_PRE_FIXTURES_ONLY=1")

        add_executable(strix-hip-gdn-retained-window-tests tests/hip_gdn_retained_window_tests.hip)
        set_target_properties(strix-hip-gdn-retained-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-gdn-retained-window-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-gdn-retained-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-retained-window-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-retained-window-tests)
        add_test(NAME strix-hip-gdn-retained-window-tests COMMAND strix-hip-gdn-retained-window-tests)

        add_executable(strix-hip-gdn-deferred-window-tests tests/hip_gdn_deferred_window_tests.hip)
        set_target_properties(strix-hip-gdn-deferred-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-gdn-deferred-window-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-gdn-deferred-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-deferred-window-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-deferred-window-tests)
        add_test(NAME strix-hip-gdn-deferred-window-tests COMMAND strix-hip-gdn-deferred-window-tests)

        add_executable(strix-hip-gdn-wide-projection-tests tests/hip_gdn_wide_projection_tests.hip)
        set_target_properties(strix-hip-gdn-wide-projection-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-gdn-wide-projection-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-gdn-wide-projection-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-wide-projection-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-wide-projection-tests)
        add_test(NAME strix-hip-gdn-wide-projection-tests COMMAND strix-hip-gdn-wide-projection-tests)

        add_executable(strix-hip-gdn-fp16-chunk-tests tests/hip_gdn_fp16_chunk_tests.hip)
        set_target_properties(strix-hip-gdn-fp16-chunk-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_compile_options(strix-hip-gdn-fp16-chunk-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-fp16-chunk-tests)
        add_test(NAME strix-hip-gdn-fp16-chunk-tests COMMAND strix-hip-gdn-fp16-chunk-tests)
        set_tests_properties(strix-hip-gdn-fp16-chunk-tests PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(strix-hip-q8-rows-projection-tests tests/hip_q8_rows_projection_tests.hip)
        set_target_properties(strix-hip-q8-rows-projection-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-q8-rows-projection-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-q8-rows-projection-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-q8-rows-projection-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-q8-rows-projection-tests)
        add_test(NAME strix-hip-q8-rows-projection-tests COMMAND strix-hip-q8-rows-projection-tests)

        # External fixtures pin compressed bits and identify the materialized
        # reference separately. Never register a GPU invocation with CTest.
        add_executable(strix-hip-dense-trellis-tests tests/hip_dense_trellis_tests.hip)
        set_target_properties(strix-hip-dense-trellis-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-dense-trellis-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-dense-trellis-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-dense-trellis-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-dense-trellis-tests)

        # Dense A8 profile prototypes: dot4 decode form vs iu8 WMMA prefill form (header-only, standalone).
        add_executable(strix-hip-dense-a8-wmma-tests tests/hip_dense_a8_wmma_tests.hip)
        set_target_properties(strix-hip-dense-a8-wmma-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(strix-hip-dense-a8-wmma-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        # hip_q8_rows_gemv.hpp reads the Q6P32 registry (strix/q6p_registry.hpp).
        target_link_libraries(strix-hip-dense-a8-wmma-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-dense-a8-wmma-tests)
        add_test(NAME strix-hip-dense-a8-wmma-tests COMMAND strix-hip-dense-a8-wmma-tests)
        add_executable(strix-hip-gsq-dense-a8-tests tests/hip_gsq_dense_a8_tests.hip)
        set_target_properties(strix-hip-gsq-dense-a8-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(strix-hip-gsq-dense-a8-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-gsq-dense-a8-tests PRIVATE strix_hip_runtime strix_core)
        strix_target_warnings(strix-hip-gsq-dense-a8-tests)
        add_test(NAME strix-hip-gsq-dense-a8-tests COMMAND strix-hip-gsq-dense-a8-tests)
        # The engine's own GSQ typed site kernels (compiled in strix_hip_runtime with its flags) through the launchers'
        # diagnostic probes: every row alone == batched T = 1..8 == the host contract.
        add_executable(strix-hip-gsq-typed-sites-tests tests/hip_gsq_typed_sites_tests.hip)
        set_target_properties(strix-hip-gsq-typed-sites-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_definitions(strix-hip-gsq-typed-sites-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gsq-typed-sites-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-gsq-typed-sites-tests PRIVATE strix_hip_runtime strix_core)
        strix_target_warnings(strix-hip-gsq-typed-sites-tests)
        add_test(NAME strix-hip-gsq-typed-sites-tests COMMAND strix-hip-gsq-typed-sites-tests)
        add_test(NAME strix-hip-prefill-dense-schedule-tests COMMAND ${CMAKE_COMMAND} -E env
            STRIX_INTPREFILL_GATES=1 $<TARGET_FILE:strix-hip-dense-a8-wmma-tests>)
        # Whole-T register-blocked dense FP16 projections (STRIX_QPF2K_DENSE=1) == launch_dense_f16, bitwise.
        add_executable(strix-hip-dense-f16-tiled-tests tests/hip_dense_f16_tiled_tests.hip)
        set_target_properties(strix-hip-dense-f16-tiled-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(strix-hip-dense-f16-tiled-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-dense-f16-tiled-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-dense-f16-tiled-tests)
        add_test(NAME strix-hip-dense-f16-tiled-tests COMMAND strix-hip-dense-f16-tiled-tests)

        if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
            add_executable(strix-hip-moe-q4k-a8-known-rows-tests tests/hip_moe_q4k_a8_known_rows_tests.hip)
            set_target_properties(strix-hip-moe-q4k-a8-known-rows-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_link_libraries(strix-hip-moe-q4k-a8-known-rows-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-moe-q4k-a8-known-rows-tests PRIVATE
                -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-q4k-a8-known-rows-tests)
            foreach(rows 2 4 8)
                add_test(NAME strix-hip-moe-q4k-a8-known-rows-${rows}
                    COMMAND strix-hip-moe-q4k-a8-known-rows-tests ${rows})
            endforeach()
            add_executable(strix-hip-moe-known-rows-tests tests/hip_moe_known_rows_tests.hip)
            set_target_properties(strix-hip-moe-known-rows-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_link_libraries(strix-hip-moe-known-rows-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-moe-known-rows-tests PRIVATE
                -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-known-rows-tests)
            foreach(rows 2 4 8)
                foreach(batched 0 1)
                    add_test(NAME strix-hip-moe-known-rows-${rows}-batched-${batched}
                        COMMAND strix-hip-moe-known-rows-tests ${rows} ${batched})
                endforeach()
                # A8 profile at canonical geometry: Q5_1, Q8_0 (layer 4), Q5_K+Q8_0 (layer 2).
                foreach(profile a8q51 a8q80 a8q5k)
                    add_test(NAME strix-hip-moe-known-rows-${rows}-${profile}
                        COMMAND strix-hip-moe-known-rows-tests ${rows} 1 ${profile})
                endforeach()
            endforeach()
            # Cohort unions of several sessions' 8-row windows (A8 profile).
            foreach(rows 12 16 24 32)
                foreach(profile a8q51 a8q80 a8q5k)
                    add_test(NAME strix-hip-moe-known-rows-${rows}-${profile}
                        COMMAND strix-hip-moe-known-rows-tests ${rows} 1 ${profile})
                endforeach()
            endforeach()
        endif()

        add_executable(strix-hip-event-queue-trace-tests tests/hip_event_queue_trace_tests.hip)
        set_target_properties(strix-hip-event-queue-trace-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-event-queue-trace-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-event-queue-trace-tests PRIVATE -mno-wavefrontsize64)
        strix_target_warnings(strix-hip-event-queue-trace-tests)
        add_test(NAME hip-event-queue-trace COMMAND strix-hip-event-queue-trace-tests)
        add_executable(strix-hip-known-window-tests tests/hip_known_window_tests.hip)
        set_target_properties(strix-hip-known-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-known-window-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-known-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-known-window-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-known-window-tests)
        add_test(NAME strix-hip-known-window-tests COMMAND strix-hip-known-window-tests)
        add_executable(strix-hip-token-sampling-device-tests tests/hip_token_sampling_device_tests.hip)
        set_target_properties(strix-hip-token-sampling-device-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-token-sampling-device-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-token-sampling-device-tests PRIVATE -mno-wavefrontsize64)
        strix_target_warnings(strix-hip-token-sampling-device-tests)
        add_test(NAME strix-hip-token-sampling-device-tests COMMAND strix-hip-token-sampling-device-tests)
        add_test(NAME strix-hip-known-window-hc-persistent-tests COMMAND strix-hip-known-window-tests --hc-rows)
        set_tests_properties(strix-hip-known-window-hc-persistent-tests PROPERTIES ENVIRONMENT "STRIX_HC_PERSISTENT=1;STRIX_HC_PERSISTENT_INJECT=1")
        add_test(NAME strix-hip-known-window-fold-tests COMMAND strix-hip-known-window-tests --window-folds)
        # STRIX_DISPATCH_FUSE=1 alone: persistent HC at 2 and 4 rows, the inject + norm fold at 8 rows.
        add_test(NAME strix-hip-known-window-hc-fuse-tests COMMAND strix-hip-known-window-tests --hc-rows)
        set_tests_properties(strix-hip-known-window-hc-fuse-tests PROPERTIES ENVIRONMENT "STRIX_DISPATCH_FUSE=1")
        add_test(NAME strix-hip-known-window-fold-fuse-tests COMMAND strix-hip-known-window-tests --window-folds)
        set_tests_properties(strix-hip-known-window-fold-fuse-tests PROPERTIES ENVIRONMENT "STRIX_DISPATCH_FUSE=1")
        # FP32 streams: the folds and the persistent HC against the separate launches.
        add_test(NAME strix-hip-known-window-fp32-fold-tests COMMAND strix-hip-known-window-tests --folds-fp32)
        set_tests_properties(strix-hip-known-window-fp32-fold-tests PROPERTIES ENVIRONMENT "STRIX_HC_PERSISTENT=1;STRIX_HC_PERSISTENT_INJECT=1")
        add_executable(strix-hip-qsa-committed-pool-tests tests/hip_qsa_committed_pool_tests.hip)
        set_target_properties(strix-hip-qsa-committed-pool-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-qsa-committed-pool-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-qsa-committed-pool-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-qsa-committed-pool-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off)
        strix_target_warnings(strix-hip-qsa-committed-pool-tests)
        add_test(NAME strix-hip-qsa-committed-pool-tests COMMAND strix-hip-qsa-committed-pool-tests)
        foreach(yarn_factor 2 4)
            add_test(NAME strix-hip-qsa-committed-pool-yarn${yarn_factor}-tests
                COMMAND strix-hip-qsa-committed-pool-tests --yarn-factor ${yarn_factor})
            set_tests_properties(strix-hip-qsa-committed-pool-yarn${yarn_factor}-tests
                PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
        endforeach()
        set_tests_properties(strix-hip-qsa-committed-pool-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-compact-stage-images-tests tests/hip_compact_stage_images_tests.hip)
        set_target_properties(strix-hip-compact-stage-images-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-compact-stage-images-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-compact-stage-images-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-compact-stage-images-tests)
        add_test(NAME strix-hip-compact-stage-images-tests COMMAND strix-hip-compact-stage-images-tests)
        set_tests_properties(strix-hip-compact-stage-images-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-qsa-decode-pool-bf16-tests tests/qsa_decode_pool_bf16.hip)
        set_target_properties(strix-hip-qsa-decode-pool-bf16-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-qsa-decode-pool-bf16-tests PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_compile_options(strix-hip-qsa-decode-pool-bf16-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-qsa-decode-pool-bf16-tests)
        add_test(NAME strix-hip-qsa-decode-pool-bf16-tests COMMAND strix-hip-qsa-decode-pool-bf16-tests)
        set_tests_properties(strix-hip-qsa-decode-pool-bf16-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-qsa-block-select-tests tests/hip_qsa_block_select_tests.hip)
        set_target_properties(strix-hip-qsa-block-select-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-qsa-block-select-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-qsa-block-select-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-qsa-block-select-tests PRIVATE -mno-wavefrontsize64)
        strix_target_warnings(strix-hip-qsa-block-select-tests)
        add_test(NAME strix-hip-qsa-block-select-tests COMMAND strix-hip-qsa-block-select-tests)
        set_tests_properties(strix-hip-qsa-block-select-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-attention-kv-int8-tests tests/hip_attention_kv_int8_tests.hip)
        set_target_properties(strix-hip-attention-kv-int8-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-attention-kv-int8-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-attention-kv-int8-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-attention-kv-int8-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math)
        strix_target_warnings(strix-hip-attention-kv-int8-tests)
        add_test(NAME strix-hip-attention-kv-int8-tests COMMAND strix-hip-attention-kv-int8-tests)
        set_tests_properties(strix-hip-attention-kv-int8-tests PROPERTIES SKIP_RETURN_CODE 77)

        # Full-model qualification is explicit and requires a model and teacher;
        # it is intentionally absent from the automatic device regression.
        add_executable(strix-hip-known-window-model-qualification tests/hip_known_window_model_qualification.hip)
        set_target_properties(strix-hip-known-window-model-qualification PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-known-window-model-qualification PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-known-window-model-qualification)

        # Explicit known-prompt/cache/continuation qualification; requires the
        # real model and ROOT-sealed inputs, so it is not an automatic CTest.
        # Explicit model-dependent serving qualification; never automatic CTest.
        # Explicit local native stage component with a real model. This does
        # not run transport or qualify a distributed model; never automatic CTest.
        add_executable(strix-hip-distributed-stage-model-qualification tests/hip_distributed_stage_tests.hip)
        set_target_properties(strix-hip-distributed-stage-model-qualification PROPERTIES
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_options(strix-hip-distributed-stage-model-qualification PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-distributed-stage-model-qualification PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-distributed-stage-model-qualification)

        # Callable model fixture requires injected actual peer/link ownership.
        # No standalone bootstrap or automatic model/transport execution.
        add_library(strix-hip-distributed-sequence-model-fixture STATIC
            tests/hip_distributed_sequence_model_qualification.hip)
        set_target_properties(strix-hip-distributed-sequence-model-fixture PROPERTIES
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_options(strix-hip-distributed-sequence-model-fixture PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-distributed-sequence-model-fixture PUBLIC strix_hip_runtime)
        target_include_directories(strix-hip-distributed-sequence-model-fixture PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        strix_target_warnings(strix-hip-distributed-sequence-model-fixture)

        # Callable actual native generation fixture; no synthetic bootstrap,
        # main function or automatic model/transport execution.
        add_library(strix-hip-distributed-greedy-model-fixture STATIC
            tests/hip_distributed_greedy_model_qualification.hip)
        set_target_properties(strix-hip-distributed-greedy-model-fixture PROPERTIES
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_options(strix-hip-distributed-greedy-model-fixture PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_include_directories(strix-hip-distributed-greedy-model-fixture PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries(strix-hip-distributed-greedy-model-fixture PUBLIC
            strix_hip_runtime strix-hip-distributed-sequence-model-fixture)
        strix_target_warnings(strix-hip-distributed-greedy-model-fixture)

        add_executable(strix-hip-serving-greedy-model-qualification tests/hip_serving_greedy_backend_tests.hip)
        set_target_properties(strix-hip-serving-greedy-model-qualification PROPERTIES
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_definitions(strix-hip-serving-greedy-model-qualification PRIVATE STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC=1)
        target_compile_options(strix-hip-serving-greedy-model-qualification PRIVATE -fno-fast-math -ffp-contract=off)
        target_link_libraries(strix-hip-serving-greedy-model-qualification PRIVATE strix_hip_runtime Threads::Threads)
        strix_target_warnings(strix-hip-serving-greedy-model-qualification)

        add_executable(strix-hip-known-prompt-model-qualification tests/hip_known_prompt_model_qualification.hip)
        set_target_properties(strix-hip-known-prompt-model-qualification PROPERTIES
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_link_libraries(strix-hip-known-prompt-model-qualification PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-known-prompt-model-qualification)

        add_executable(
            strix-hip-moe-experts-launcher-tests
            tests/hip_moe_experts_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-moe-experts-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-moe-experts-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-moe-experts-launcher-tests
        )
        add_test(
            NAME strix-hip-moe-experts-launcher-tests
            COMMAND strix-hip-moe-experts-launcher-tests
        )

        if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
            add_executable(strix-hip-q51-grouped-down-probe-tests tests/hip_q51_grouped_down_probe_tests.hip)
            add_test(NAME strix-hip-moe-a8-down-runtime-admission-tests
                COMMAND strix-hip-moe-kquant-tests 4 a8-down-runtime-admission)
            add_test(NAME strix-hip-moe-r1-glue-tail-tests COMMAND strix-hip-moe-kquant-tests 0 r1-glue-tail)
            set_target_properties(strix-hip-q51-grouped-down-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_link_libraries(strix-hip-q51-grouped-down-probe-tests PRIVATE strix_hip_runtime)
            strix_target_warnings(strix-hip-q51-grouped-down-probe-tests)
            add_test(NAME strix-hip-q51-grouped-down-probe-tests COMMAND strix-hip-q51-grouped-down-probe-tests)
            add_executable(strix-hip-q51-slot-parallel-probe-tests tests/hip_q51_slot_parallel_probe_tests.hip)
            set_target_properties(strix-hip-q51-slot-parallel-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_link_libraries(strix-hip-q51-slot-parallel-probe-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-q51-slot-parallel-probe-tests PRIVATE -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q51-slot-parallel-probe-tests)
            add_test(NAME strix-hip-q51-slot-parallel-probe-tests COMMAND strix-hip-q51-slot-parallel-probe-tests)
            foreach(mode 0 1 4 5)
                add_test(NAME strix-hip-moe-q51-slot-parallel-native-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-slot-parallel)
            endforeach()
            foreach(mode 0 4)
                add_test(NAME strix-hip-moe-q51-slot-parallel-native-duplicate-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-slot-parallel-duplicate)
            endforeach()
            foreach(mode 0 1 4 5)
                add_test(NAME strix-hip-moe-q51-grouped-native-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-grouped)
            endforeach()
            foreach(mode 0 4)
                add_test(NAME strix-hip-moe-q51-grouped-native-duplicate-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-grouped-duplicate)
            endforeach()
            foreach(mode 0 4)
                add_test(NAME strix-hip-moe-q4k-a8-native-duplicate-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-duplicate)
            endforeach()
            foreach(mode 0 1 4 5)
                add_test(NAME strix-hip-moe-q4k-a8-native-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8)
            endforeach()
        endif()
        add_test(NAME strix-q4k-a8-native-cli-tests
            COMMAND "${CMAKE_COMMAND}" "-DBINARY=$<TARGET_FILE:strix-hip-decode-token>"
            "-DCAPABILITY=${STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_q4k_a8_runtime_cli.cmake")
        add_executable(strix-hip-moe-kquant-tests tests/hip_moe_kquant_tests.hip)
        set_target_properties(strix-hip-moe-kquant-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-moe-kquant-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-kquant-tests)
        foreach(mode RANGE 0 7)
            add_test(NAME strix-hip-moe-kquant-${mode}-tests
                     COMMAND strix-hip-moe-kquant-tests ${mode})
        endforeach()

        add_executable(strix-hip-ggml-weight-formats-tests tests/hip_ggml_weight_formats_tests.hip)
        set_target_properties(strix-hip-ggml-weight-formats-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-ggml-weight-formats-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-ggml-weight-formats-tests)
        add_test(NAME strix-hip-ggml-weight-formats-tests COMMAND strix-hip-ggml-weight-formats-tests)

        add_executable(strix-hip-gdn-output-projection-tests tests/hip_gdn_output_projection_tests.hip)
        target_link_libraries(strix-hip-gdn-output-projection-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-gdn-output-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        strix_target_warnings(strix-hip-gdn-output-projection-tests)
        add_test(NAME strix-hip-gdn-output-projection-tests COMMAND strix-hip-gdn-output-projection-tests)

        add_executable(strix-hip-qsa-index-batch-tests tests/hip_qsa_index_batch_tests.hip)
        target_link_libraries(strix-hip-qsa-index-batch-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-qsa-index-batch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        strix_target_warnings(strix-hip-qsa-index-batch-tests)
        add_test(NAME strix-hip-qsa-index-batch-tests COMMAND strix-hip-qsa-index-batch-tests 7)
        set_tests_properties(strix-hip-qsa-index-batch-tests PROPERTIES ENVIRONMENT "STRIX_QSA_INDEX_ROWS=1;STRIX_QPF2K_INDEX_BATCH=1;STRIX_PREFILL_INDEX_PROJ_WMMA=1;STRIX_PREFILL_EXACT_FP16=0")

        add_executable(strix-hip-qsa-output-projection-tests tests/hip_qsa_output_projection_tests.hip)
        target_compile_definitions(strix-hip-qsa-output-projection-tests PRIVATE STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC=1)
        target_link_libraries(strix-hip-qsa-output-projection-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-qsa-output-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        strix_target_warnings(strix-hip-qsa-output-projection-tests)
        add_test(NAME strix-hip-qsa-output-projection-tests COMMAND strix-hip-qsa-output-projection-tests)

        add_executable(
            strix-hip-gdn-launcher-tests
            tests/hip_gdn_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-gdn-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-gdn-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-gdn-launcher-tests
        )
        add_test(
            NAME strix-hip-gdn-launcher-tests
            COMMAND strix-hip-gdn-launcher-tests
        )

        add_test(NAME strix-hip-gdn-virtual-q8-f32-fallback-tests
            COMMAND strix-hip-gdn-launcher-tests 128 fallback)
        add_test(NAME strix-hip-gdn-prefetch-q8-f32-fallback-tests
            COMMAND strix-hip-gdn-launcher-tests 128 prefetch-fallback)
        foreach(prefetch_width 128 288 480 640 2560)
            add_test(NAME strix-hip-gdn-prefetch-q8-${prefetch_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${prefetch_width} prefetch)
        endforeach()
        foreach(q8_width 128 288 640 2560)
            add_test(NAME strix-hip-gdn-virtual-q8-${q8_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${q8_width})
        endforeach()

        add_executable(
            strix-hip-ple-launcher-tests
            tests/hip_ple_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-ple-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-ple-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-ple-launcher-tests
        )
        add_test(
            NAME strix-hip-ple-launcher-tests
            COMMAND strix-hip-ple-launcher-tests
        )
        # STRIX_PLE_WINDOW_WARM: verify-window PLE rows with and without decode cache warming, bitwise.
        add_executable(strix-hip-ple-window-warm-tests tests/hip_ple_window_warm_tests.hip)
        set_target_properties(strix-hip-ple-window-warm-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-ple-window-warm-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-ple-window-warm-tests)
        add_test(NAME strix-hip-ple-window-warm-tests COMMAND strix-hip-ple-window-warm-tests)

        add_executable(strix-hip-qsa-input-prefetch-tests tests/hip_qsa_input_prefetch_tests.hip)
        set_target_properties(strix-hip-qsa-input-prefetch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-qsa-input-prefetch-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-input-prefetch-tests)
        add_test(NAME strix-hip-qsa-input-prefetch-tests COMMAND strix-hip-qsa-input-prefetch-tests)
        add_executable(strix-hip-shared-expert-prefetch-tests tests/hip_shared_expert_prefetch_tests.hip)
        set_target_properties(strix-hip-shared-expert-prefetch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-shared-expert-prefetch-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-shared-expert-prefetch-tests)
        add_test(NAME strix-hip-shared-expert-prefetch-tests COMMAND strix-hip-shared-expert-prefetch-tests)

        add_executable(strix-hip-q8-projection-organization-tests tests/hip_q8_projection_organization_tests.hip)
        set_target_properties(strix-hip-q8-projection-organization-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-q8-projection-organization-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-q8-projection-organization-tests)
        add_test(NAME strix-hip-q8-projection-organization-tests COMMAND strix-hip-q8-projection-organization-tests)
        foreach(wave8_width IN ITEMS 128 288 480 640 2560)
            add_test(NAME strix-hip-gdn-wave8-q8-${wave8_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${wave8_width} wave8)
        endforeach()
        foreach(tail_heads IN ITEMS 4 12)
            add_test(NAME strix-hip-gdn-wave8-q8-tail-${tail_heads}-tests
                COMMAND strix-hip-gdn-launcher-tests 2560 wave8 ${tail_heads})
        endforeach()
        foreach(wave8_output_width IN ITEMS 128 288 480 640 2560)
            add_test(NAME strix-hip-gdn-wave8-q8-output-${wave8_output_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${wave8_output_width} wave8-output)
        endforeach()
        foreach(output_heads IN ITEMS 4 12)
            add_test(NAME strix-hip-gdn-wave8-q8-output-heads-${output_heads}-tests
                COMMAND strix-hip-gdn-launcher-tests 640 wave8-output ${output_heads})
        endforeach()
        add_executable(strix-hip-gdn-lm-wave8-diagnostic-tests tests/hip_gdn_lm_wave8_diagnostic_tests.hip)
        add_executable(strix-hip-shared-expert-rows-diagnostic-tests tests/hip_shared_expert_rows_diagnostic_tests.hip)
        set_target_properties(strix-hip-shared-expert-rows-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-shared-expert-rows-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-shared-expert-rows-diagnostic-tests)
        add_test(NAME strix-hip-shared-expert-rows-diagnostic-tests COMMAND strix-hip-shared-expert-rows-diagnostic-tests)
        add_executable(strix-hip-qpersist-attention-tests tests/hip_qpersist_attention_tests.hip)
        set_target_properties(strix-hip-qpersist-attention-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-qpersist-attention-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qpersist-attention-tests)
        add_test(NAME strix-hip-qpersist-attention-tests COMMAND strix-hip-qpersist-attention-tests)
        add_executable(strix-hip-router-select-tests tests/hip_router_select_tests.hip)
        set_target_properties(strix-hip-router-select-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-router-select-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-router-select-tests)
        add_test(NAME strix-hip-router-select-tests COMMAND strix-hip-router-select-tests 5)
        add_executable(strix-hip-draft-attention-exact-tests tests/hip_draft_attention_exact_tests.hip)
        set_target_properties(strix-hip-draft-attention-exact-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-draft-attention-exact-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-draft-attention-exact-tests)
        add_test(NAME strix-hip-draft-attention-exact-tests COMMAND strix-hip-draft-attention-exact-tests)
        add_executable(strix-hip-qsa-window-attn-v2-tests tests/hip_qsa_window_attn_v2_tests.hip)
        set_target_properties(strix-hip-qsa-window-attn-v2-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-qsa-window-attn-v2-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-window-attn-v2-tests)
        add_test(NAME strix-hip-qsa-window-attn-v2-tests COMMAND strix-hip-qsa-window-attn-v2-tests)
        add_executable(strix-hip-qsa-gqa-reuse-diagnostic-tests tests/hip_qsa_gqa_reuse_diagnostic_tests.hip)
        set_target_properties(strix-hip-qsa-gqa-reuse-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-qsa-gqa-reuse-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-gqa-reuse-diagnostic-tests)
        add_test(NAME strix-hip-qsa-gqa-reuse-diagnostic-tests COMMAND strix-hip-qsa-gqa-reuse-diagnostic-tests)
        add_executable(strix-hip-moe-batched-activation-diagnostic-tests tests/hip_moe_batched_activation_diagnostic_tests.hip)
        set_target_properties(strix-hip-moe-batched-activation-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-moe-batched-activation-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-batched-activation-diagnostic-tests)
        add_test(NAME strix-hip-moe-batched-activation-diagnostic-tests COMMAND strix-hip-moe-batched-activation-diagnostic-tests)
        add_executable(strix-hip-moe-overlap-dispatcher-tests tests/hip_moe_overlap_dispatcher_tests.hip)
        set_target_properties(strix-hip-moe-overlap-dispatcher-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-moe-overlap-dispatcher-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-overlap-dispatcher-tests)
        add_test(NAME strix-hip-moe-overlap-dispatcher-tests COMMAND strix-hip-moe-overlap-dispatcher-tests)
        add_executable(strix-hip-generation-output-tests tests/hip_generation_output_tests.hip)
        add_executable(strix-hip-gdn-recurrent-layout-diagnostic-tests tests/hip_gdn_recurrent_layout_diagnostic_tests.hip)
        set_target_properties(strix-hip-gdn-recurrent-layout-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-gdn-recurrent-layout-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-gdn-recurrent-layout-diagnostic-tests)
        add_test(NAME strix-hip-gdn-recurrent-layout-diagnostic-tests COMMAND strix-hip-gdn-recurrent-layout-diagnostic-tests)
        set_target_properties(strix-hip-generation-output-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-generation-output-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-generation-output-tests)
        add_test(NAME strix-hip-generation-output-tests COMMAND strix-hip-generation-output-tests)
        set_target_properties(strix-hip-gdn-lm-wave8-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_link_libraries(strix-hip-gdn-lm-wave8-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-gdn-lm-wave8-diagnostic-tests)
        add_test(NAME strix-hip-gdn-lm-wave8-diagnostic-tests COMMAND strix-hip-gdn-lm-wave8-diagnostic-tests)

        add_executable(
            strix-hip-indexed-attention-launcher-tests
            tests/hip_indexed_attention_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-indexed-attention-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}"
        )
        target_link_libraries(
            strix-hip-indexed-attention-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-indexed-attention-launcher-tests
        )
        add_test(
            NAME strix-hip-indexed-attention-launcher-tests
            COMMAND strix-hip-indexed-attention-launcher-tests
        )

        add_executable(
            strix-hip-model-residency-tests
            tests/hip_model_residency_tests.cpp
            src/hip_model_residency.cpp
        )
        target_compile_definitions(strix-hip-model-residency-tests PRIVATE STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC=1)
        target_link_libraries(
            strix-hip-model-residency-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-model-residency-tests
        )
        add_test(
            NAME strix-hip-model-residency-tests
            COMMAND strix-hip-model-residency-tests
        )

        add_executable(
            strix-hip-session-snapshot-transfer-tests
            tests/hip_session_snapshot_transfer_tests.cpp
        )
        target_link_libraries(
            strix-hip-session-snapshot-transfer-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-session-snapshot-transfer-tests
        )
        add_test(
            NAME strix-hip-session-snapshot-transfer-tests
            COMMAND strix-hip-session-snapshot-transfer-tests
        )

        add_executable(
            strix-hip-runtime-memory-init-tests
            tests/hip_runtime_memory_init_tests.cpp
        )
        target_link_libraries(
            strix-hip-runtime-memory-init-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-runtime-memory-init-tests
        )
        add_test(
            NAME strix-hip-runtime-memory-init-tests
            COMMAND strix-hip-runtime-memory-init-tests
        )
    endif()
endif()

if(STRIX_ENABLE_HIP_RUNTIME)
    # Two-rank distributed runtime: transports and the device-buffer communicator.
    add_library(strix_dist STATIC src/dist_communicator.cpp src/dist_gpu_direct.cpp src/dist_gpu_direct_setup.cpp
        src/dist_gpu_direct_copy.hip src/dist_host_copy.hip src/dist_staged_ready.hip src/linux_gpu_queue.cpp)
    target_include_directories(strix_dist PUBLIC include)
    target_include_directories(strix_dist SYSTEM PUBLIC "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_features(strix_dist PUBLIC cxx_std_20)
    target_compile_definitions(strix_dist PUBLIC $<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__=1>)
    find_package(Threads REQUIRED)
    target_link_libraries(strix_dist PUBLIC "${STRIX_HIP_RUNTIME_LIBRARY}" Threads::Threads PRIVATE strix_serve)
    strix_target_warnings(strix_dist)
    add_executable(strix-dist-tcp-probe apps/dist_tcp_probe_main.cpp)
    target_link_libraries(strix-dist-tcp-probe PRIVATE strix_dist)
    strix_target_warnings(strix-dist-tcp-probe)
    add_executable(strix-dist-host-exchange-bench apps/dist_host_exchange_bench_main.cpp)
    target_link_libraries(strix-dist-host-exchange-bench PRIVATE strix_dist)
    strix_target_warnings(strix-dist-host-exchange-bench)
    add_executable(strix-dist-gpu-probe apps/dist_gpu_probe_main.cpp)
    target_link_libraries(strix-dist-gpu-probe PRIVATE strix_dist)
    strix_target_warnings(strix-dist-gpu-probe)
    if(BUILD_TESTING)
        add_executable(strix-dist-transport-tests tests/dist_transport_tests.cpp)
        add_executable(strix-dist-crc32c-tests tests/dist_crc32c_tests.cpp)
        target_compile_features(strix-dist-crc32c-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-dist-crc32c-tests)
        add_test(NAME strix-dist-crc32c-tests COMMAND strix-dist-crc32c-tests)
        add_executable(strix-hip-dist-hostpath-tests tests/hip_dist_hostpath_tests.cpp)
        target_link_libraries(strix-hip-dist-hostpath-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-hostpath-tests)
        target_link_libraries(strix-dist-transport-tests PRIVATE strix_dist)
        strix_target_warnings(strix-dist-transport-tests)
        add_test(NAME strix-dist-transport-tests COMMAND strix-dist-transport-tests)
        add_executable(strix-hip-dist-gpu-direct-tests tests/hip_dist_gpu_direct_tests.cpp)
        add_executable(strix-hip-dist-host-copy-tests tests/hip_dist_host_copy_tests.cpp)
        add_executable(strix-hip-staged-ready-tests tests/hip_staged_ready_tests.cpp)
        target_link_libraries(strix-hip-staged-ready-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-staged-ready-tests)
        add_executable(strix-hip-dist-crc-defer-tests tests/hip_dist_crc_defer_tests.cpp)
        target_link_libraries(strix-hip-dist-crc-defer-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-crc-defer-tests)
        target_link_libraries(strix-hip-dist-host-copy-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-host-copy-tests)
        target_link_libraries(strix-hip-dist-gpu-direct-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-gpu-direct-tests)
    endif()
endif()
endif()  # NOT STRIX_CUDA_DS41_ONLY

# ---- deepseek41 model family module (optional) ----
if(STRIX_FAMILY_DEEPSEEK41)
add_library(strix_family_deepseek41 STATIC src/deepseek41_model.cpp src/deepseek41_plan.cpp src/deepseek41_family.cpp)
target_link_libraries(strix_family_deepseek41 PUBLIC strix_core)
strix_target_warnings(strix_family_deepseek41)
add_executable(strix-deepseek41-inspect apps/deepseek41_inspect_main.cpp)
target_link_libraries(strix-deepseek41-inspect PRIVATE strix_family_deepseek41)
strix_target_warnings(strix-deepseek41-inspect)
if(BUILD_TESTING)
    add_executable(strix-deepseek41-plan-tests tests/deepseek41_plan_tests.cpp)
    target_link_libraries(strix-deepseek41-plan-tests PRIVATE strix_family_deepseek41)
    strix_target_warnings(strix-deepseek41-plan-tests)
    add_test(NAME strix-deepseek41-plan-tests COMMAND strix-deepseek41-plan-tests)
    add_executable(strix-deepseek41-model-tests tests/deepseek41_model_tests.cpp)
    target_link_libraries(strix-deepseek41-model-tests PRIVATE strix_family_deepseek41)
    strix_target_warnings(strix-deepseek41-model-tests)
    add_test(NAME strix-deepseek41-model-tests COMMAND strix-deepseek41-model-tests)
endif()
# deepseek41 weight/cache formats: isolated host and device TUs with strict arithmetic flags.
add_library(strix_ds41_formats STATIC src/ds41_formats.cpp src/ds41_moe.cpp src/ds41_packed.cpp src/ds41_expert_tp.cpp)
target_include_directories(strix_ds41_formats PUBLIC include)
target_compile_features(strix_ds41_formats PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_formats)
if(NOT MSVC)
    target_compile_options(strix_ds41_formats PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-formats-tests tests/ds41_formats_tests.cpp)
    target_link_libraries(strix-ds41-formats-tests PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-ds41-formats-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-formats-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-formats-tests COMMAND strix-ds41-formats-tests)
    add_executable(strix-ds41-expert-tp-tests tests/ds41_expert_tp_tests.cpp)
    target_link_libraries(strix-ds41-expert-tp-tests PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-ds41-expert-tp-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-expert-tp-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-expert-tp-tests COMMAND strix-ds41-expert-tp-tests)
    add_executable(strix-ds41-packed-tests tests/ds41_packed_tests.cpp)
    target_link_libraries(strix-ds41-packed-tests PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-ds41-packed-tests)
    add_test(NAME strix-ds41-packed-tests COMMAND strix-ds41-packed-tests)
endif()
# deepseek41 native FP8 dense weights of the original checkpoint: safetensors reader and the exact product reference.
add_library(strix_ds41_native_fp8 STATIC src/ds41_native_fp8.cpp)
target_link_libraries(strix_ds41_native_fp8 PUBLIC strix_ds41_formats strix_serve_json)
strix_target_warnings(strix_ds41_native_fp8)
if(NOT MSVC)
    target_compile_options(strix_ds41_native_fp8 PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)  # strix_serve is not configured there
    add_executable(strix-ds41-native-fp8-tests tests/ds41_native_fp8_tests.cpp)
    target_link_libraries(strix-ds41-native-fp8-tests PRIVATE strix_ds41_native_fp8)
    strix_target_warnings(strix-ds41-native-fp8-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-native-fp8-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-native-fp8-tests COMMAND strix-ds41-native-fp8-tests)
endif()
# deepseek41 trellis dense sites (ds41_trellis.hpp, identity ds41-trellis-k5-v1): the pack reader and slicer (host).
add_library(strix_ds41_trellis STATIC src/ds41_trellis.cpp)
target_link_libraries(strix_ds41_trellis PUBLIC strix_ds41_native_fp8)
strix_target_warnings(strix_ds41_trellis)
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)  # strix_serve is not configured there
    # site bytes, step / configuration identities under single packs and mixes, the weights namespace (host only)
    add_executable(strix-ds41-trellis-identity-tests tests/ds41_trellis_identity_tests.cpp)
    target_link_libraries(strix-ds41-trellis-identity-tests PRIVATE strix_ds41_trellis)
    strix_target_warnings(strix-ds41-trellis-identity-tests)
    add_test(NAME strix-ds41-trellis-identity-tests COMMAND strix-ds41-trellis-identity-tests)
endif()
# deepseek41 hot-expert placement: reader of the versioned placement file and the local/remote split of a layer's
# routed rows under the canonical combine (host only).
add_library(strix_ds41_hot_placement STATIC src/ds41_hot_placement.cpp)
target_include_directories(strix_ds41_hot_placement PUBLIC include)
target_compile_features(strix_ds41_hot_placement PUBLIC cxx_std_20)
target_link_libraries(strix_ds41_hot_placement PUBLIC strix_serve_json)
strix_target_warnings(strix_ds41_hot_placement)
if(BUILD_TESTING)
    add_executable(strix-ds41-hot-placement-tests tests/ds41_hot_placement_tests.cpp)
    target_link_libraries(strix-ds41-hot-placement-tests PRIVATE strix_ds41_hot_placement)
    strix_target_warnings(strix-ds41-hot-placement-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-hot-placement-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-hot-placement-tests COMMAND strix-ds41-hot-placement-tests)
endif()
# deepseek41 layer sites (mHC, norms, RoPE, projection sites, embedding/head, Engram): host reference
# and the Engram hash and file reader, with the same strict arithmetic flags as the formats.
add_library(strix_ds41_layer STATIC src/ds41_layer.cpp src/ds41_engram.cpp)
target_include_directories(strix_ds41_layer PUBLIC include)
target_compile_features(strix_ds41_layer PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(strix_ds41_layer PUBLIC strix_ds41_formats strix_family_deepseek41 Threads::Threads)
strix_target_warnings(strix_ds41_layer)
if(NOT MSVC)
    target_compile_options(strix_ds41_layer PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-layer-tests tests/ds41_layer_tests.cpp)
    target_link_libraries(strix-ds41-layer-tests PRIVATE strix_ds41_layer)
    strix_target_warnings(strix-ds41-layer-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-layer-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-layer-tests COMMAND strix-ds41-layer-tests)
    add_executable(strix-ds41-engram-reader-bench tests/ds41_engram_reader_bench.cpp)
    target_link_libraries(strix-ds41-engram-reader-bench PRIVATE strix_ds41_layer)
    strix_target_warnings(strix-ds41-engram-reader-bench)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(strix-ds41-engram-cache-tests tests/ds41_engram_cache_tests.cpp)
        target_link_libraries(strix-ds41-engram-cache-tests PRIVATE strix_ds41_layer)
        strix_target_warnings(strix-ds41-engram-cache-tests)
        add_test(NAME strix-ds41-engram-cache-tests COMMAND strix-ds41-engram-cache-tests)
        add_executable(strix-ds41-engram-trace-bench tests/ds41_engram_trace_bench.cpp)
        target_link_libraries(strix-ds41-engram-trace-bench PRIVATE strix_ds41_layer)
        strix_target_warnings(strix-ds41-engram-trace-bench)
    endif()
endif()
# deepseek41 decode attention: isolated host references and device kernels with strict arithmetic flags.
add_library(strix_ds41_attention STATIC src/ds41_attention.cpp)
target_include_directories(strix_ds41_attention PUBLIC include)
target_link_libraries(strix_ds41_attention PUBLIC strix_family_deepseek41)
target_compile_features(strix_ds41_attention PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_attention)
if(NOT MSVC)
    target_compile_options(strix_ds41_attention PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-attention-tests tests/ds41_attention_tests.cpp)
    target_link_libraries(strix-ds41-attention-tests PRIVATE strix_ds41_attention strix_ds41_formats)
    strix_target_warnings(strix-ds41-attention-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-attention-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-attention-tests COMMAND strix-ds41-attention-tests)
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_attention_hip STATIC src/hip_ds41_attention.hip)
    target_link_libraries(strix_ds41_attention_hip PUBLIC strix_ds41_attention "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_attention_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    set(strix_ds41_attention_hip_targets strix_ds41_attention_hip)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-attention-tests tests/hip_ds41_attention_tests.hip)
        target_link_libraries(strix-hip-ds41-attention-tests PRIVATE strix_ds41_attention_hip strix_ds41_hip)
        add_test(NAME strix-hip-ds41-attention-tests COMMAND strix-hip-ds41-attention-tests)
        list(APPEND strix_ds41_attention_hip_targets strix-hip-ds41-attention-tests)
        # FAST lever ds41-attn-f16wmma-v1: error bounds vs the exact attention + timing of both
        add_executable(strix-hip-ds41-fast-attn-tests tests/hip_ds41_fast_attn_tests.hip)
        target_link_libraries(strix-hip-ds41-fast-attn-tests PRIVATE strix_ds41_attention_hip)
        add_test(NAME strix-hip-ds41-fast-attn-tests COMMAND strix-hip-ds41-fast-attn-tests --no-timing)
        list(APPEND strix_ds41_attention_hip_targets strix-hip-ds41-fast-attn-tests)
        # timing build: the same kernels with per-block clocks (STRIX_DS41_ATTN_TIMING), not a test
        add_executable(strix-hip-ds41-attention-timing tests/hip_ds41_attention_timing.hip src/hip_ds41_attention.hip)
        target_link_libraries(strix-hip-ds41-attention-timing PRIVATE strix_ds41_attention "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix-hip-ds41-attention-timing PRIVATE include "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_definitions(strix-hip-ds41-attention-timing PRIVATE STRIX_DS41_ATTN_TIMING)
        list(APPEND strix_ds41_attention_hip_targets strix-hip-ds41-attention-timing)
    endif()
    foreach(target ${strix_ds41_attention_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(${target} PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(${target})
    endforeach()
endif()
# deepseek41 trimmed-model host reference (integration): plan interpreter over the formats/MoE pieces.
add_library(strix_ds41_runtime STATIC src/deepseek41_host_forward.cpp src/deepseek41_attention_host.cpp src/ds41_weight_overlay.cpp
    src/deepseek41_attention_state.cpp src/deepseek41_prefix_cache.cpp src/deepseek41_pair_defaults.cpp)
target_link_libraries(strix_ds41_runtime PUBLIC strix_ds41_layer strix_ds41_formats strix_ds41_attention strix_family_deepseek41)
strix_target_warnings(strix_ds41_runtime)
if(NOT MSVC)
    target_compile_options(strix_ds41_runtime PRIVATE -ffp-contract=off)
endif()
# Session compatibility uses checkpoint loaders and serving JSON, outside the CUDA replica.
if(NOT STRIX_CUDA_DS41_ONLY)
    add_library(strix_ds41_session STATIC src/deepseek41_session_cache.cpp)
    target_link_libraries(strix_ds41_session PUBLIC strix_ds41_runtime strix_ds41_trellis strix_ds41_native_fp8)
    strix_target_warnings(strix_ds41_session)
    if(NOT MSVC)
        target_compile_options(strix_ds41_session PRIVATE -ffp-contract=off)
    endif()
endif()
# Versioned offline rank-local expert storage, shared by decode and batch.
add_library(strix_ds41_prepared_load STATIC src/ds41_prepared_load.cpp)
target_include_directories(strix_ds41_prepared_load PUBLIC include)
target_compile_features(strix_ds41_prepared_load PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(strix_ds41_prepared_load PUBLIC strix_ds41_runtime Threads::Threads)
# Prepared payload verification must not silently use a slow scalar SHA when the
# accelerated backend is installed. This leaves the full-file hashing option unchanged.
if(NOT TARGET OpenSSL::Crypto)
    find_package(OpenSSL QUIET COMPONENTS Crypto)
endif()
if(TARGET OpenSSL::Crypto)
    target_compile_definitions(strix_ds41_prepared_load PRIVATE STRIX_PREPARED_OPENSSL=1)
    target_link_libraries(strix_ds41_prepared_load PRIVATE OpenSSL::Crypto)
    message(STATUS "Prepared-load chunk SHA-256: OpenSSL (v1 full verification)")
else()
    message(STATUS "Prepared-load chunk SHA-256: portable (v1 full verification, parallel chunks)")
endif()
strix_target_warnings(strix_ds41_prepared_load)
add_executable(strix-ds41-prepare-load apps/ds41_prepare_load_main.cpp)
target_link_libraries(strix-ds41-prepare-load PRIVATE strix_ds41_prepared_load)
strix_target_warnings(strix-ds41-prepare-load)
if(BUILD_TESTING)
    add_executable(strix-ds41-prepared-load-tests tests/ds41_prepared_load_tests.cpp)
    target_link_libraries(strix-ds41-prepared-load-tests PRIVATE strix_ds41_prepared_load)
    strix_target_warnings(strix-ds41-prepared-load-tests)
    add_test(NAME strix-ds41-prepared-load-tests COMMAND strix-ds41-prepared-load-tests)
endif()
add_executable(strix-deepseek41-trimmed apps/deepseek41_trimmed_main.cpp)
target_link_libraries(strix-deepseek41-trimmed PRIVATE strix_ds41_runtime)
strix_target_warnings(strix-deepseek41-trimmed)
# attention-state handoff over TCP (prefill host -> decode rank 0), with --synth states of the released plan
add_executable(strix-ds41-state-send apps/ds41_state_send_main.cpp)
target_link_libraries(strix-ds41-state-send PRIVATE strix_ds41_runtime)
target_compile_definitions(strix-ds41-state-send PRIVATE STRIX_DS41_STATE_SYNTH=1)
if(WIN32)
    target_link_libraries(strix-ds41-state-send PRIVATE ws2_32)
endif()
strix_target_warnings(strix-ds41-state-send)
if(BUILD_TESTING)
    add_executable(strix-deepseek41-attention-state-tests tests/deepseek41_attention_state_tests.cpp)
    target_link_libraries(strix-deepseek41-attention-state-tests PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-deepseek41-attention-state-tests)
    add_test(NAME strix-deepseek41-attention-state-tests COMMAND strix-deepseek41-attention-state-tests)
endif()
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)
    add_executable(strix-serve-session-disabled-tests tests/serve_session_disabled_tests.cpp)
    target_link_libraries(strix-serve-session-disabled-tests PRIVATE strix_serve)
    target_sources(strix-serve-session-disabled-tests PRIVATE src/deepseek41_pair_defaults.cpp)
    strix_target_warnings(strix-serve-session-disabled-tests)
    add_test(NAME strix-serve-session-disabled-tests COMMAND strix-serve-session-disabled-tests --session-cache off)
    add_test(NAME strix-serve-session-enabled-tests COMMAND strix-serve-session-disabled-tests --enabled)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-serve-session-no-session-identity-tests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/serve_session_defaults_compare.py $<TARGET_FILE:strix-serve-session-disabled-tests>)
    endif()
    add_executable(strix-deepseek41-session-cache-tests tests/deepseek41_session_cache_tests.cpp src/ds41_file_reader.cpp)
    target_link_libraries(strix-deepseek41-session-cache-tests PRIVATE strix_ds41_session strix_ds41_dspark Threads::Threads)
    strix_target_warnings(strix-deepseek41-session-cache-tests)
    add_test(NAME strix-deepseek41-session-cache-tests COMMAND strix-deepseek41-session-cache-tests)
    set_tests_properties(strix-deepseek41-session-cache-tests PROPERTIES TIMEOUT 120)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-deepseek41-session-cache-restart-tests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/deepseek41_session_cache_restart_test.py $<TARGET_FILE:strix-deepseek41-session-cache-tests>)
        set_tests_properties(strix-deepseek41-session-cache-restart-tests PROPERTIES TIMEOUT 120)
        find_program(STRIX_SESSION_TEST_BASH bash)
        if(STRIX_SESSION_TEST_BASH)
            add_test(NAME strix-deepseek41-session-cache-gate-tests COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/deepseek41_session_cache_gate_tests.py)
        endif()
    endif()
endif()
if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-deepseek41-prefix-probe-tests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/ds41_prefix_cache_probe_tests.py)
    endif()
    add_executable(strix-deepseek41-serve-cache-policy-tests tests/deepseek41_serve_cache_policy_tests.cpp)
    target_include_directories(strix-deepseek41-serve-cache-policy-tests PRIVATE include)
    target_compile_features(strix-deepseek41-serve-cache-policy-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-deepseek41-serve-cache-policy-tests)
    add_test(NAME strix-deepseek41-serve-cache-policy-tests COMMAND strix-deepseek41-serve-cache-policy-tests)
    add_executable(strix-deepseek41-prefix-cache-tests tests/deepseek41_prefix_cache_tests.cpp)
    target_link_libraries(strix-deepseek41-prefix-cache-tests PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-deepseek41-prefix-cache-tests)
    add_test(NAME strix-deepseek41-prefix-cache-tests COMMAND strix-deepseek41-prefix-cache-tests)
    add_executable(strix-deepseek41-pair-defaults-tests tests/deepseek41_pair_defaults_tests.cpp)
    target_link_libraries(strix-deepseek41-pair-defaults-tests PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-deepseek41-pair-defaults-tests)
    add_test(NAME strix-deepseek41-pair-defaults-tests COMMAND strix-deepseek41-pair-defaults-tests)
endif()
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)
    add_executable(strix-deepseek41-serve-sched-tests tests/deepseek41_serve_sched_tests.cpp)
    target_link_libraries(strix-deepseek41-serve-sched-tests PRIVATE strix_ds41_runtime strix_serve)
    strix_target_warnings(strix-deepseek41-serve-sched-tests)
    add_test(NAME strix-deepseek41-serve-sched-tests COMMAND strix-deepseek41-serve-sched-tests)
    add_executable(strix-deepseek41-serve-observe-tests tests/deepseek41_serve_observe_tests.cpp)
    target_include_directories(strix-deepseek41-serve-observe-tests PRIVATE include)
    target_compile_features(strix-deepseek41-serve-observe-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-deepseek41-serve-observe-tests PRIVATE strix_serve)
    strix_target_warnings(strix-deepseek41-serve-observe-tests)
    add_test(NAME strix-deepseek41-serve-observe-tests COMMAND strix-deepseek41-serve-observe-tests)
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_hip STATIC src/hip_ds41_formats.hip src/hip_ds41_routes_fast.hip src/hip_ds41_expert_tp.hip)
    target_link_libraries(strix_ds41_hip PUBLIC strix_ds41_formats "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-formats-tests tests/hip_ds41_formats_tests.hip)
        target_link_libraries(strix-hip-ds41-formats-tests PRIVATE strix_ds41_hip strix_dist strix_core)
        add_test(NAME strix-hip-ds41-formats-tests COMMAND strix-hip-ds41-formats-tests)
        set(strix_ds41_hip_targets strix_ds41_hip strix-hip-ds41-formats-tests)
    else()
        set(strix_ds41_hip_targets strix_ds41_hip)
    endif()
    add_library(strix_ds41_layer_hip STATIC src/hip_ds41_layer.hip src/hip_ds41_f16_rows.hip src/hip_ds41_rn_wide.hip src/hip_ds41_rn_table.hip src/hip_ds41_f8_rows_fast.hip)
    target_link_libraries(strix_ds41_layer_hip PUBLIC strix_ds41_hip strix_ds41_layer strix_ds41_prefill_hip)
    target_include_directories(strix_ds41_layer_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    list(APPEND strix_ds41_hip_targets strix_ds41_layer_hip)
    option(STRIX_DS41_PREFILL_FP16_LT "Build optional FP16 dense prefill backend" OFF)
    if(STRIX_DS41_PREFILL_FP16_LT)
        find_library(STRIX_HIPBLASLT_LIBRARY NAMES hipblaslt PATHS /opt/rocm/lib /opt/rocm/core-10.1/lib REQUIRED)
        target_sources(strix_ds41_layer_hip PRIVATE src/hip_ds41_dense_fp16.hip)
        target_link_libraries(strix_ds41_layer_hip PUBLIC "${STRIX_HIPBLASLT_LIBRARY}")
        get_filename_component(strix_hipblaslt_real "${STRIX_HIPBLASLT_LIBRARY}" REALPATH)
        get_filename_component(strix_hipblaslt_dir "${strix_hipblaslt_real}" DIRECTORY)
        target_link_options(strix_ds41_layer_hip INTERFACE "-Wl,-rpath,${strix_hipblaslt_dir}")

        set_property(SOURCE src/hip_deepseek41_prefill.hip APPEND PROPERTY COMPILE_DEFINITIONS STRIX_DS41_PREFILL_FP16_LT=1)
        add_executable(strix-hip-ds41-approx-prefill-tests tests/hip_ds41_approx_prefill_tests.hip)
        target_link_libraries(strix-hip-ds41-approx-prefill-tests PRIVATE strix_ds41_layer_hip strix_ds41_attention_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-approx-prefill-tests)

    endif()
    # Trellis dense sites (ds41_trellis.hpp): device side on the external trellis library (C ABI 1, a static archive
    # checked by its SHA256). No default location: pass -DSTRIX_DS41_TRELLIS_LIBRARY=<archive> and
    # -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=<dir with strix_trellis.h>. ON by default only where both are given and exist;
    # OFF builds a stub that refuses STRIX_DS41_TRELLIS at load time, so every other build keeps working.
    set(STRIX_DS41_TRELLIS_LIBRARY "" CACHE FILEPATH "Trellis library archive (with the dense-r1 and grouped GEMV APIs for the default dense_r1 adapter)")
    set(STRIX_DS41_TRELLIS_INCLUDE_DIR "" CACHE PATH "Directory with strix_trellis.h")
    if(STRIX_DS41_TRELLIS_LIBRARY AND STRIX_DS41_TRELLIS_INCLUDE_DIR AND EXISTS "${STRIX_DS41_TRELLIS_LIBRARY}"
       AND EXISTS "${STRIX_DS41_TRELLIS_INCLUDE_DIR}/strix_trellis.h")
        set(strix_ds41_trellis_default ON)
    else()
        set(strix_ds41_trellis_default OFF)
    endif()
    option(STRIX_DS41_TRELLIS "Link the trellis library for the deepseek41 dense sites" ${strix_ds41_trellis_default})
    set(STRIX_DS41_TRELLIS_API "dense_r1" CACHE STRING "Trellis adapter: dense_r1 (qualified production identity), v12, tp03, splitk04")
    set_property(CACHE STRIX_DS41_TRELLIS_API PROPERTY STRINGS v12 tp03 splitk04 dense_r1)
    if(NOT STRIX_DS41_TRELLIS_API MATCHES "^(v12|tp03|splitk04|dense_r1)$")
        message(FATAL_ERROR "Unknown STRIX_DS41_TRELLIS_API=${STRIX_DS41_TRELLIS_API}")
    endif()
    add_library(strix_ds41_trellis_hip STATIC src/hip_ds41_trellis.hip src/hip_ds41_trellis_group.hip src/hip_ds41_shexp_tp.hip)
    target_link_libraries(strix_ds41_trellis_hip PUBLIC strix_ds41_trellis strix_ds41_formats "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_trellis_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    if(STRIX_DS41_TRELLIS)
        if(NOT EXISTS "${STRIX_DS41_TRELLIS_LIBRARY}" OR NOT EXISTS "${STRIX_DS41_TRELLIS_INCLUDE_DIR}/strix_trellis.h")
            message(FATAL_ERROR "STRIX_DS41_TRELLIS=ON needs STRIX_DS41_TRELLIS_LIBRARY (archive) and "
                                "STRIX_DS41_TRELLIS_INCLUDE_DIR (with strix_trellis.h)")
        endif()
        if(NOT STRIX_DS41_TRELLIS_API STREQUAL "v12")
            target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_PREPARED=1)
        endif()
        if(STRIX_DS41_TRELLIS_API MATCHES "^(splitk04|dense_r1)$")
            target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_SPLITK=1)
        endif()
        if(STRIX_DS41_TRELLIS_API STREQUAL "dense_r1")
            target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_DENSE_R1=1)
            # the apps' production defaults (deepseek41_pair_defaults.hpp) put the trellis packs on only with this API
            target_compile_definitions(strix_ds41_trellis_hip PUBLIC STRIX_DS41_PAIR_DENSE_R1=1)
        endif()
        message(STATUS "deepseek41 trellis: ${STRIX_DS41_TRELLIS_LIBRARY}")
        target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_LIB=1)
        target_include_directories(strix_ds41_trellis_hip SYSTEM PUBLIC "${STRIX_DS41_TRELLIS_INCLUDE_DIR}")
        target_link_libraries(strix_ds41_trellis_hip PUBLIC "${STRIX_DS41_TRELLIS_LIBRARY}")
    endif()
    list(APPEND strix_ds41_hip_targets strix_ds41_trellis_hip)
    if(BUILD_TESTING AND STRIX_DS41_TRELLIS)
        # R-invariance of the library rows (GEMV R = 1..8, GEMM, the chunked linear) on real pack tensors; run explicitly on a GPU
        add_executable(strix-hip-ds41-trellis-tests tests/hip_ds41_trellis_tests.hip)
        target_link_libraries(strix-hip-ds41-trellis-tests PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-tests)
        # Persistent split-K counters: changing layouts and generic GEMV must not overwrite another plan's counters.
        # Explicit invocation on a GPU only; no CTest registration.
        add_executable(strix-hip-ds41-trellis-workspace-tests tests/hip_ds41_trellis_workspace_tests.hip)
        target_link_libraries(strix-hip-ds41-trellis-workspace-tests PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-workspace-tests)
        # trellis LM head adapter (STRIX_DS41_TRELLIS_HEADOUT): in-place output / padding vs today's staging; run explicitly on a GPU
        add_executable(strix-hip-ds41-trellis-headout-bench tests/hip_ds41_trellis_headout_bench.hip)
        target_link_libraries(strix-hip-ds41-trellis-headout-bench PRIVATE strix_ds41_trellis_hip strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-headout-bench)
        # TrellisGroupRunner's grouped-library path against a shim of the requested grouped strided GEMV; run explicitly on a GPU
        add_executable(strix-hip-ds41-trellis-group-shim-tests tests/hip_ds41_trellis_group_shim_tests.hip)
        target_link_libraries(strix-hip-ds41-trellis-group-shim-tests PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-group-shim-tests)
        # shared expert TP on the trellis library (ds41-trellis-k5-v1:shexp-tp): exactness of the lean chain, cold timing
        add_executable(strix-hip-ds41-shexp-tp-bench tests/hip_ds41_shexp_tp_bench.hip)
        target_link_libraries(strix-hip-ds41-shexp-tp-bench PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-shexp-tp-bench)
    endif()
    # trimmed-model device forward (P4) and the host/device lockstep gate
    add_library(strix_ds41_device_forward STATIC src/hip_deepseek41_device_forward.hip)
    target_link_libraries(strix_ds41_device_forward PUBLIC strix_ds41_prepared_load strix_ds41_layer_hip strix_ds41_attention_hip strix_ds41_runtime strix_dist
        strix_ds41_native_fp8 strix_ds41_pipeline strix_ds41_trellis_hip)
    target_include_directories(strix_ds41_device_forward PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    list(APPEND strix_ds41_hip_targets strix_ds41_device_forward)
    add_executable(strix-deepseek41-lockstep apps/deepseek41_lockstep_main.cpp)
    target_link_libraries(strix-deepseek41-lockstep PRIVATE strix_ds41_device_forward)
    strix_target_warnings(strix-deepseek41-lockstep)
    # one rank of the two-node decode (expert halves + split head over the communicator), or both ranks in-process
    add_executable(strix-deepseek41-rank apps/deepseek41_rank_main.cpp)
    target_link_libraries(strix-deepseek41-rank PRIVATE strix_ds41_device_forward strix_dist strix_ds41_pipeline_front strix_ds41_session)
    strix_target_warnings(strix-deepseek41-rank)
    # the FRONT of a pipeline (layers [0, K) here, the rest on a strix-deepseek41-rank --role pipeline-stage TAIL)
    add_executable(strix-deepseek41-pipeline-front apps/deepseek41_pipeline_front_main.cpp)
    target_link_libraries(strix-deepseek41-pipeline-front PRIVATE strix_ds41_device_forward strix_ds41_pipeline_front)
    strix_target_warnings(strix-deepseek41-pipeline-front)
    # requantization quality gate: teacher-forced windows, reference slices and per-row statistics, calibration sums
    add_executable(strix-deepseek41-requant-eval apps/deepseek41_requant_eval_main.cpp)
    target_link_libraries(strix-deepseek41-requant-eval PRIVATE strix_ds41_device_forward strix_dist)
    strix_target_warnings(strix-deepseek41-requant-eval)
    # several requests per forward step (BatchForward), its gate/measurement app and the DeepSeek HTTP server
    add_library(strix_ds41_batch_forward STATIC src/hip_deepseek41_batch_forward.hip)
    target_link_libraries(strix_ds41_batch_forward PUBLIC strix_ds41_prepared_load strix_ds41_layer_hip strix_ds41_attention_hip strix_ds41_runtime strix_dist strix_ds41_trellis_hip strix_ds41_pipeline)
    target_include_directories(strix_ds41_batch_forward PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    list(APPEND strix_ds41_hip_targets strix_ds41_batch_forward)
    add_executable(strix-deepseek41-batch apps/deepseek41_batch_main.cpp)
    target_link_libraries(strix-deepseek41-batch PRIVATE strix_ds41_batch_dspark strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-batch)
    add_executable(strix-deepseek41-serve apps/deepseek41_serve_main.cpp src/handoff.cpp src/bp_codec.cpp)  # --import-frames
    target_link_libraries(strix-deepseek41-serve PRIVATE strix_ds41_batch_dspark strix_ds41_batch_forward strix_ds41_device_prefill strix_dist strix_serve strix_core strix_ds41_pipeline strix_ds41_session)
    strix_target_warnings(strix-deepseek41-serve)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-layer-tests tests/hip_ds41_layer_tests.hip)
        add_executable(strix-hip-ds41-r1fuse-tests tests/hip_ds41_r1fuse_tests.hip)
        target_link_libraries(strix-hip-ds41-r1fuse-tests PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-r1fuse-tests)
        add_test(NAME strix-hip-ds41-r1fuse-tests COMMAND strix-hip-ds41-r1fuse-tests)
        add_executable(strix-hip-ds41-gaps-bench tests/hip_ds41_gaps_bench.hip)
        target_link_libraries(strix-hip-ds41-gaps-bench PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-gaps-bench)
        add_executable(strix-hip-ds41-ready-tests tests/hip_ds41_ready_tests.hip)
        target_link_libraries(strix-hip-ds41-ready-tests PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-ready-tests)
        add_executable(strix-hip-ds41-ready-bench tests/hip_ds41_ready_bench.hip)
        target_link_libraries(strix-hip-ds41-ready-bench PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-ready-bench)
        target_link_libraries(strix-hip-ds41-layer-tests PRIVATE strix_ds41_layer_hip)
        add_test(NAME strix-hip-ds41-layer-tests COMMAND strix-hip-ds41-layer-tests)
        add_executable(strix-hip-ds41-f16-rows-tests tests/hip_ds41_f16_rows_tests.hip)
        target_link_libraries(strix-hip-ds41-f16-rows-tests PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        add_test(NAME strix-hip-ds41-f16-rows-tests COMMAND strix-hip-ds41-f16-rows-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-f16-rows-tests)
        add_executable(strix-hip-ds41-engram-f16-pf-tests tests/hip_ds41_engram_f16_pf_tests.hip)
        target_link_libraries(strix-hip-ds41-engram-f16-pf-tests PRIVATE strix_ds41_layer_hip)
        add_test(NAME strix-hip-ds41-engram-f16-pf-tests COMMAND strix-hip-ds41-engram-f16-pf-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-engram-f16-pf-tests)
        add_executable(strix-hip-ds41-layer-bench tests/hip_ds41_layer_bench.hip)
        target_link_libraries(strix-hip-ds41-layer-bench PRIVATE strix_ds41_layer_hip)
        add_executable(strix-hip-ds41-rn-wide-bench tests/hip_ds41_rn_wide_bench.hip)
        target_link_libraries(strix-hip-ds41-rn-wide-bench PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-rn-wide-bench)
        # trellis LM head bench: only with an external strix_trellis static library (-DSTRIX_DS41_TRELLIS_HEAD_BENCH_LIB=<.a>
        # -DSTRIX_DS41_TRELLIS_HEAD_BENCH_INCLUDE=<dir of strix_trellis.h>)
        if(STRIX_DS41_TRELLIS_HEAD_BENCH_LIB)
            add_executable(strix-hip-ds41-trellis-head-bench tests/hip_ds41_trellis_head_bench.hip)
            target_include_directories(strix-hip-ds41-trellis-head-bench PRIVATE "${STRIX_DS41_TRELLIS_HEAD_BENCH_INCLUDE}")
            target_link_libraries(strix-hip-ds41-trellis-head-bench PRIVATE strix_ds41_layer_hip strix_ds41_formats
                "${STRIX_DS41_TRELLIS_HEAD_BENCH_LIB}")
        endif()
        add_executable(strix-hip-ds41-rn-sel-bench tests/hip_ds41_rn_sel_bench.hip)
        target_link_libraries(strix-hip-ds41-rn-sel-bench PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-rn-sel-bench)
        # Trellis attention projections bench (needs the trellis library, see STRIX_DS41_TRELLIS above).
        if(STRIX_DS41_TRELLIS)
            add_executable(strix-hip-ds41-trellis-attn-bench tests/hip_ds41_trellis_attn_bench.hip)
            target_include_directories(strix-hip-ds41-trellis-attn-bench SYSTEM PRIVATE "${STRIX_DS41_TRELLIS_INCLUDE_DIR}")
            target_link_libraries(strix-hip-ds41-trellis-attn-bench PRIVATE strix_ds41_layer_hip "${STRIX_DS41_TRELLIS_LIBRARY}")
            # o_a adapter (per-group calls vs the grouped runner, STRIX_DS41_TRELLIS_OA_GROUP) on the real pack, + today's U6
            add_executable(strix-hip-ds41-trellis-oa-bench tests/hip_ds41_trellis_oa_bench.hip)
            target_link_libraries(strix-hip-ds41-trellis-oa-bench PRIVATE strix_ds41_trellis_hip strix_ds41_layer_hip)
        endif()
        add_executable(strix-hip-ds41-engram-bench tests/hip_ds41_engram_bench.hip)
        target_link_libraries(strix-hip-ds41-engram-bench PRIVATE strix_ds41_layer_hip)
        add_executable(strix-hip-ds41-engram-kv-bench tests/hip_ds41_engram_kv_bench.hip)
        target_link_libraries(strix-hip-ds41-engram-kv-bench PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        add_executable(strix-hip-ds41-layer-prefill-bench tests/hip_ds41_layer_prefill_bench.hip)
        target_link_libraries(strix-hip-ds41-layer-prefill-bench PRIVATE strix_ds41_layer_hip)
        # native FP8 x A8 sites and the shared-expert SwiGLU + A8 epilogue against product_fp8 / quantize_a8, bitwise
        add_executable(strix-hip-ds41-native-fp8-tests tests/hip_ds41_native_fp8_tests.hip)
        target_link_libraries(strix-hip-ds41-native-fp8-tests PRIVATE strix_ds41_layer_hip strix_ds41_native_fp8)
        add_test(NAME strix-hip-ds41-native-fp8-tests COMMAND strix-hip-ds41-native-fp8-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-native-fp8-tests)
        add_executable(strix-hip-ds41-f16-wide-tests tests/hip_ds41_f16_wide_tests.hip)
        target_link_libraries(strix-hip-ds41-f16-wide-tests PRIVATE strix_ds41_layer_hip)
        add_test(NAME strix-hip-ds41-f16-wide-tests COMMAND strix-hip-ds41-f16-wide-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-layer-tests strix-hip-ds41-layer-bench strix-hip-ds41-engram-bench
            strix-hip-ds41-engram-kv-bench
            strix-hip-ds41-layer-prefill-bench strix-hip-ds41-f16-wide-tests)
        # routed/shared expert and head kernels at the runner's launch shapes: timings and form checks
        add_executable(strix-hip-ds41-moe-bench tests/hip_ds41_moe_bench.hip)
        target_link_libraries(strix-hip-ds41-moe-bench PRIVATE strix_ds41_layer_hip strix_core)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-moe-bench)
        add_executable(strix-hip-ds41-expert-once-bench tests/hip_ds41_expert_once_bench.hip)
        target_link_libraries(strix-hip-ds41-expert-once-bench PRIVATE strix_ds41_layer_hip strix_core)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-expert-once-bench)
        # router fast form, paired F16 projections, compress ahead: against the current kernels (STRIX_DS41_SMALL_OPS_BENCH=1: timings)
        add_executable(strix-hip-ds41-small-ops-tests tests/hip_ds41_small_ops_tests.hip)
        target_link_libraries(strix-hip-ds41-small-ops-tests PRIVATE strix_ds41_layer_hip strix_ds41_attention_hip strix_ds41_formats)
        add_test(NAME strix-hip-ds41-small-ops-tests COMMAND strix-hip-ds41-small-ops-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-small-ops-tests)
    endif()
    foreach(target ${strix_ds41_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(${target} PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(${target})
    endforeach()
endif()

# deepseek41 prefill: routed experts of T rows (WMMA iu8, decode-identical arithmetic) and the exact
# suffix replay plan.
add_library(strix_ds41_prefill STATIC src/ds41_prefill.cpp)
target_include_directories(strix_ds41_prefill PUBLIC include)
target_compile_features(strix_ds41_prefill PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_prefill)
if(BUILD_TESTING)
    add_executable(strix-ds41-prefill-tests tests/ds41_prefill_tests.cpp)
    target_link_libraries(strix-ds41-prefill-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-prefill-tests)
    add_test(NAME strix-ds41-prefill-tests COMMAND strix-ds41-prefill-tests)
    add_executable(strix-ds41-prefill-tuning-tests tests/ds41_prefill_tuning_tests.cpp)
    target_link_libraries(strix-ds41-prefill-tuning-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-prefill-tuning-tests)
    add_test(NAME strix-ds41-prefill-tuning-tests COMMAND strix-ds41-prefill-tuning-tests)
    add_executable(strix-ds41-host-pin-admission-tests tests/ds41_host_pin_admission_tests.cpp)
    target_link_libraries(strix-ds41-host-pin-admission-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-host-pin-admission-tests)
    add_test(NAME strix-ds41-host-pin-admission-tests COMMAND strix-ds41-host-pin-admission-tests)
    add_executable(strix-ds41-host-pin-plan-tests tests/ds41_host_pin_plan_tests.cpp)
    target_link_libraries(strix-ds41-host-pin-plan-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-host-pin-plan-tests)
    add_test(NAME strix-ds41-host-pin-plan-tests COMMAND strix-ds41-host-pin-plan-tests)
    add_executable(strix-ds41-host-promote-tests tests/ds41_host_promote_tests.cpp)
    target_link_libraries(strix-ds41-host-promote-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-host-promote-tests)
    add_test(NAME strix-ds41-host-promote-tests COMMAND strix-ds41-host-promote-tests)
endif()

# The pipeline wire (serving front end, TCP) and the DSpark drafter are not part of the CUDA-only configure.
if(NOT STRIX_CUDA_DS41_ONLY)
# deepseek41 heterogeneous pipeline (portable): the strix/ds41-pipeline-v1 wire between pipeline stages (messages, the
# cut's side sections, TCP) and the tensormeld/engine-describe-v1 description generated from the model file header.
add_library(strix_ds41_pipeline STATIC src/ds41_pipeline_wire.cpp src/ds41_pipeline_multi.cpp src/ds41_pipeline_dspark.cpp src/deepseek41_describe.cpp)
target_link_libraries(strix_ds41_pipeline PUBLIC strix_family_deepseek41 strix_ds41_prefill strix_serve_json)
strix_target_warnings(strix_ds41_pipeline)
# the engine revision --describe-json reports: -DSTRIX_SOURCE_REVISION=..., else the tree's .source-revision file
if(NOT STRIX_SOURCE_REVISION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.source-revision")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/.source-revision" STRIX_SOURCE_REVISION LIMIT_COUNT 1)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/.source-revision")
endif()
if(STRIX_SOURCE_REVISION)
    set_source_files_properties(src/deepseek41_describe.cpp PROPERTIES COMPILE_DEFINITIONS "STRIX_SOURCE_REVISION=\"${STRIX_SOURCE_REVISION}\"")
endif()
# the FRONT side (backend interface, TAIL client, state manifest)
add_library(strix_ds41_pipeline_front STATIC src/ds41_pipeline_front.cpp)
target_link_libraries(strix_ds41_pipeline_front PUBLIC strix_ds41_pipeline strix_ds41_runtime)
strix_target_warnings(strix_ds41_pipeline_front)
# a FRONT without layers: replays recorded STEP payloads to a TAIL (exactness and TAIL timing without a FRONT's compute)
add_executable(strix-deepseek41-pipeline-replay apps/deepseek41_pipeline_replay_main.cpp)
target_link_libraries(strix-deepseek41-pipeline-replay PRIVATE strix_ds41_pipeline)
strix_target_warnings(strix-deepseek41-pipeline-replay)
if(BUILD_TESTING)
    if(NOT WIN32)  # pipeline Connection/Listener implement POSIX sockets only
        add_executable(strix-ds41-pipeline-wire-tests tests/ds41_pipeline_wire_tests.cpp)
        target_link_libraries(strix-ds41-pipeline-wire-tests PRIVATE strix_ds41_pipeline_front)
        strix_target_warnings(strix-ds41-pipeline-wire-tests)
        add_test(NAME strix-ds41-pipeline-wire-tests COMMAND strix-ds41-pipeline-wire-tests)
        add_executable(strix-ds41-pipeline-multi-tests tests/ds41_pipeline_multi_tests.cpp)
        target_link_libraries(strix-ds41-pipeline-multi-tests PRIVATE strix_ds41_pipeline)
        strix_target_warnings(strix-ds41-pipeline-multi-tests)
        add_test(NAME strix-ds41-pipeline-multi-tests COMMAND strix-ds41-pipeline-multi-tests)
    endif()
    add_executable(strix-ds41-pipeline-dspark-tests tests/ds41_pipeline_dspark_tests.cpp)
    target_link_libraries(strix-ds41-pipeline-dspark-tests PRIVATE strix_ds41_pipeline)
    strix_target_warnings(strix-ds41-pipeline-dspark-tests)
    add_executable(strix-deepseek41-describe-tests tests/deepseek41_describe_tests.cpp)
    target_link_libraries(strix-deepseek41-describe-tests PRIVATE strix_ds41_pipeline)
    strix_target_warnings(strix-deepseek41-describe-tests)
    add_test(NAME strix-deepseek41-describe-tests COMMAND strix-deepseek41-describe-tests)
    # length routing of the prefill (local two-node prefill vs offload + state handoff; header only)
    add_executable(strix-ds41-prefill-route-tests tests/ds41_prefill_route_tests.cpp)
    target_link_libraries(strix-ds41-prefill-route-tests PRIVATE strix_serve_json)
    strix_target_warnings(strix-ds41-prefill-route-tests)
    add_test(NAME strix-ds41-prefill-route-tests COMMAND strix-ds41-prefill-route-tests)
endif()
# deepseek41 expert server (portable part): the strix/ds41-expert-rpc-v1 wire between a parts-host and expert servers
# (POSIX or Winsock), the server's session loop (direct or relay over two ranks) and its CPU reference backend, and the
# round-trip probe. The HIP backend is below (STRIX_ENABLE_HIP_RUNTIME). Strict FP: the server folds and the reference
# combine must keep the one-device combine's bits.
add_library(strix_ds41_xrpc STATIC src/ds41_expert_rpc.cpp)
target_include_directories(strix_ds41_xrpc PUBLIC include)
target_compile_features(strix_ds41_xrpc PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_xrpc)
add_library(strix_ds41_xs STATIC src/ds41_expert_server.cpp)
target_link_libraries(strix_ds41_xs PUBLIC strix_ds41_xrpc strix_ds41_formats strix_core Threads::Threads)
strix_target_warnings(strix_ds41_xs)
add_executable(strix-ds41-xs-probe apps/ds41_xs_probe_main.cpp)
target_link_libraries(strix-ds41-xs-probe PRIVATE strix_ds41_xrpc Threads::Threads)
strix_target_warnings(strix-ds41-xs-probe)
add_executable(strix-ds41-xs-check apps/ds41_xs_check_main.cpp)
target_link_libraries(strix-ds41-xs-check PRIVATE strix_ds41_xs)
strix_target_warnings(strix-ds41-xs-check)
if(NOT STRIX_ENABLE_HIP_RUNTIME)
    add_executable(strix-deepseek41-expert-server apps/deepseek41_expert_server_main.cpp)
    target_link_libraries(strix-deepseek41-expert-server PRIVATE strix_ds41_xs)
    strix_target_warnings(strix-deepseek41-expert-server)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-expert-rpc-tests tests/ds41_expert_rpc_tests.cpp)
    target_link_libraries(strix-ds41-expert-rpc-tests PRIVATE strix_ds41_xs)
    strix_target_warnings(strix-ds41-expert-rpc-tests)
    add_test(NAME strix-ds41-expert-rpc-tests COMMAND strix-ds41-expert-rpc-tests)
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    # the HIP backend (routed experts resident on a gfx1151 node) and the server with relay over USB4
    add_library(strix_ds41_xs_hip STATIC src/hip_ds41_expert_backend.hip)
    target_link_libraries(strix_ds41_xs_hip PUBLIC strix_ds41_xs strix_ds41_hip "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_xs_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    set_target_properties(strix_ds41_xs_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
    target_compile_options(strix_ds41_xs_hip PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
        -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
    strix_target_warnings(strix_ds41_xs_hip)
    add_executable(strix-deepseek41-expert-server apps/deepseek41_expert_server_main.cpp)
    target_link_libraries(strix-deepseek41-expert-server PRIVATE strix_ds41_xs_hip strix_dist)
    target_compile_definitions(strix-deepseek41-expert-server PRIVATE STRIX_XS_WITH_HIP)
    strix_target_warnings(strix-deepseek41-expert-server)
endif()
if(NOT MSVC)
    foreach(target strix_ds41_xrpc strix_ds41_xs)
        target_compile_options(${target} PRIVATE -ffp-contract=off)
    endforeach()
    target_compile_options(strix-ds41-xs-check PRIVATE -ffp-contract=off)
    if(BUILD_TESTING)
        target_compile_options(strix-ds41-expert-rpc-tests PRIVATE -ffp-contract=off)
    endif()
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_prefill_hip STATIC src/hip_ds41_prefill.hip)
    target_link_libraries(strix_ds41_prefill_hip PUBLIC strix_ds41_prefill strix_ds41_hip strix_ds41_formats "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_prefill_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    # Two-rank prefill of one MoE layer (expert halves, half sums exchanged per chunk).
    add_executable(strix-ds41-prefill-pair apps/ds41_prefill_pair_main.hip)
    target_link_libraries(strix-ds41-prefill-pair PRIVATE strix_ds41_prefill_hip strix_ds41_hip strix_dist strix_core)
    set(strix_ds41_prefill_hip_targets strix_ds41_prefill_hip strix-ds41-prefill-pair)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-prefill-tests tests/hip_ds41_prefill_tests.hip)
        target_link_libraries(strix-hip-ds41-prefill-tests PRIVATE strix_ds41_prefill_hip strix_ds41_hip strix_core)
        add_test(NAME strix-hip-ds41-prefill-tests COMMAND strix-hip-ds41-prefill-tests)
        # FAST lever ds41-moe-relaxed-v1: error vs the exact batch kernels + timings
        add_executable(strix-hip-ds41-fast-moe-tests tests/hip_ds41_fast_moe_tests.hip)
        target_link_libraries(strix-hip-ds41-fast-moe-tests PRIVATE strix_ds41_prefill_hip strix_ds41_hip strix_core)
        add_test(NAME strix-hip-ds41-fast-moe-tests COMMAND strix-hip-ds41-fast-moe-tests --no-timing)
        list(APPEND strix_ds41_prefill_hip_targets strix-hip-ds41-fast-moe-tests)
        list(APPEND strix_ds41_prefill_hip_targets strix-hip-ds41-prefill-tests)
        add_executable(strix-hip-ds41-prefill-tiles-tests tests/hip_ds41_prefill_tiles_tests.hip)
        target_link_libraries(strix-hip-ds41-prefill-tiles-tests PRIVATE strix_ds41_layer_hip)
        add_test(NAME strix-hip-ds41-prefill-tiles-tests COMMAND strix-hip-ds41-prefill-tiles-tests)
        list(APPEND strix_ds41_prefill_hip_targets strix-hip-ds41-prefill-tiles-tests)
    endif()
    foreach(target ${strix_ds41_prefill_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(${target} PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(${target})
    endforeach()
endif()

# deepseek41 DSpark drafter: host reference (declared arithmetic) and device kernels.
add_library(strix_ds41_dspark STATIC src/ds41_dspark.cpp src/ds41_dspark_files.cpp src/ds41_dspark_driver.cpp src/ds41_dspark_vocab.cpp)
target_link_libraries(strix_ds41_dspark PUBLIC strix_ds41_formats strix_core)
find_package(Threads REQUIRED)
target_link_libraries(strix_ds41_dspark PUBLIC Threads::Threads)
strix_target_warnings(strix_ds41_dspark)
if(NOT MSVC)
    target_compile_options(strix_ds41_dspark PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-dspark-tests tests/ds41_dspark_tests.cpp)
    target_link_libraries(strix-ds41-dspark-tests PRIVATE strix_ds41_dspark strix_core)
    strix_target_warnings(strix-ds41-dspark-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-dspark-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-dspark-tests COMMAND strix-ds41-dspark-tests)
    add_executable(strix-ds41-dspark-table-tests tests/ds41_dspark_table_tests.cpp)
    target_link_libraries(strix-ds41-dspark-table-tests PRIVATE strix_ds41_dspark strix_core)
    strix_target_warnings(strix-ds41-dspark-table-tests)
    add_test(NAME strix-ds41-dspark-table-tests COMMAND strix-ds41-dspark-table-tests)
endif()
add_executable(strix-deepseek41-dspark apps/deepseek41_dspark_main.cpp)
target_link_libraries(strix-deepseek41-dspark PRIVATE strix_ds41_runtime strix_ds41_dspark)
strix_target_warnings(strix-deepseek41-dspark)
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_dspark_hip STATIC src/hip_ds41_dspark.hip)
    target_link_libraries(strix_ds41_dspark_hip PUBLIC strix_ds41_dspark strix_ds41_hip)
    set(strix_ds41_dspark_hip_targets strix_ds41_dspark_hip)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-dspark-tests tests/hip_ds41_dspark_tests.hip)
        target_link_libraries(strix-hip-ds41-dspark-tests PRIVATE strix_ds41_dspark_hip strix_dist strix_core)
        add_test(NAME strix-hip-ds41-dspark-tests COMMAND strix-hip-ds41-dspark-tests)
        list(APPEND strix_ds41_dspark_hip_targets strix-hip-ds41-dspark-tests)
    endif()
    add_executable(strix-ds41-dspark-rank apps/ds41_dspark_rank_main.hip)
    target_link_libraries(strix-ds41-dspark-rank PRIVATE strix_ds41_dspark_hip strix_dist)
    list(APPEND strix_ds41_dspark_hip_targets strix-ds41-dspark-rank)
    # DSpark windows inside BatchForward (several requests per step): drafting contexts per request, the planner
    add_library(strix_ds41_batch_dspark STATIC src/hip_deepseek41_batch_dspark.hip)
    target_link_libraries(strix_ds41_batch_dspark PUBLIC strix_ds41_dspark_hip strix_ds41_batch_forward strix_lookup)
    list(APPEND strix_ds41_dspark_hip_targets strix_ds41_batch_dspark)
    add_executable(strix-deepseek41-dspark-rank apps/deepseek41_dspark_rank_main.hip)
    target_link_libraries(strix-deepseek41-dspark-rank PRIVATE strix_ds41_dspark_hip strix_ds41_device_forward strix_dist)
    list(APPEND strix_ds41_dspark_hip_targets strix-deepseek41-dspark-rank)
    add_executable(strix-ds41-dspark-replay apps/ds41_dspark_replay_main.hip)
    target_link_libraries(strix-ds41-dspark-replay PRIVATE strix_ds41_dspark_hip)
    list(APPEND strix_ds41_dspark_hip_targets strix-ds41-dspark-replay)
    # the drafter of a pipeline TAIL (speculative extension of strix/ds41-pipeline-v1)
    add_library(strix_ds41_spec_tail STATIC src/hip_ds41_spec_tail.hip)
    target_link_libraries(strix_ds41_spec_tail PUBLIC strix_ds41_dspark_hip)
    list(APPEND strix_ds41_dspark_hip_targets strix_ds41_spec_tail)
    if(TARGET strix-deepseek41-rank)
        target_link_libraries(strix-deepseek41-rank PRIVATE strix_ds41_spec_tail)
    endif()
    foreach(target ${strix_ds41_dspark_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
        target_compile_options(${target} PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(${target})
    endforeach()
endif()
# deepseek41 device prefill: T rows of the decode plan per layer (one node or one of two ranks) and its app.
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_device_prefill STATIC src/hip_deepseek41_prefill.hip)
    target_link_libraries(strix_ds41_device_prefill PUBLIC strix_ds41_device_forward strix_ds41_prefill_hip strix_ds41_layer_hip
        strix_ds41_attention_hip strix_dist)
    target_include_directories(strix_ds41_device_prefill PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    set_target_properties(strix_ds41_device_prefill PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
    target_compile_options(strix_ds41_device_prefill PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
        -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
    strix_target_warnings(strix_ds41_device_prefill)
    add_executable(strix-deepseek41-prefill apps/deepseek41_prefill_main.cpp)
    target_link_libraries(strix-deepseek41-prefill PRIVATE strix_ds41_device_prefill strix_ds41_device_forward strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-prefill)
    add_executable(strix-deepseek41-prefill-quality apps/deepseek41_prefill_quality_main.cpp)
    target_link_libraries(strix-deepseek41-prefill-quality PRIVATE strix_ds41_device_prefill strix_dist)
    strix_target_warnings(strix-deepseek41-prefill-quality)
    add_executable(strix-deepseek41-prefill-continue apps/deepseek41_prefill_continue_main.cpp)
    target_link_libraries(strix-deepseek41-prefill-continue PRIVATE strix_ds41_device_prefill strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-prefill-continue)

    # long context: the device prefill's attention state handed to the decode forward (or read from a file), then decode
    add_executable(strix-deepseek41-longctx apps/deepseek41_longctx_main.cpp)
    target_link_libraries(strix-deepseek41-longctx PRIVATE strix_ds41_device_prefill strix_ds41_device_forward strix_dist)
    strix_target_warnings(strix-deepseek41-longctx)
    # prefix cache gate: a prompt resumed from cached block states == its fresh prefill (tokens, logits, state digest)
    add_executable(strix-deepseek41-prefix apps/deepseek41_prefix_main.cpp)
    target_link_libraries(strix-deepseek41-prefix PRIVATE strix_ds41_device_prefill strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-prefix)
endif()
endif()  # NOT STRIX_CUDA_DS41_ONLY
# Host-forward capture for the CUDA layer-core check (host only; also built with the CUDA backend).
if(STRIX_DS41_TAP_CAPTURE OR STRIX_ENABLE_CUDA_DS41)
    add_executable(strix-ds41-tap-capture tests/cuda/ds41_core_tap_capture.cpp)
    target_link_libraries(strix-ds41-tap-capture PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-ds41-tap-capture)
endif()
# strix/ds41-expert-rpc-v1 (std only, POSIX or Winsock): the wire between a parts-host and expert servers.
# Strict FP: the reference combine must keep the one-device combine's bits. Defined once: the expert server block above
# defines it when that part of the tree is configured.
if(NOT TARGET strix_ds41_xrpc)
    add_library(strix_ds41_xrpc STATIC src/ds41_expert_rpc.cpp)
    target_include_directories(strix_ds41_xrpc PUBLIC include)
    target_compile_features(strix_ds41_xrpc PUBLIC cxx_std_20)
    strix_target_warnings(strix_ds41_xrpc)
    if(NOT MSVC)
        target_compile_options(strix_ds41_xrpc PRIVATE -ffp-contract=off)
    endif()
endif()
# deepseek41 CUDA backend (sm_120, sm_89, sm_86): layer kernels with the declared arithmetic of ds41-exact-v1,
# checked against the host references. Off by default; nothing else depends on it. The kernels use the INT8
# mma.sync / ldmatrix / cp.async these architectures share (same bits on each); select_device() refuses a device
# the build has no code for.
if(STRIX_ENABLE_CUDA_DS41)
    add_executable(strix-cuda-ds41-attention-alloc-tests tests/cuda/ds41_attention_alloc_tests.cpp)
    target_link_libraries(strix-cuda-ds41-attention-alloc-tests PRIVATE strix_ds41_attention strix_ds41_prefill)
    strix_target_warnings(strix-cuda-ds41-attention-alloc-tests)
    if(BUILD_TESTING)
        add_test(NAME strix-cuda-ds41-attention-alloc-tests COMMAND strix-cuda-ds41-attention-alloc-tests)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "120;89;86")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(STRIX_DS41_CUDA_SOURCES src/cuda/ds41_cuda_common.cu src/cuda/ds41_core.cu src/cuda/ds41_core_norm.cu
        src/cuda/ds41_core_moe.cu src/cuda/ds41_core_io.cu src/cuda/ds41_core_mma.cu src/cuda/ds41_router_v2.cu
        src/cuda/ds41_core_layer.cu)
    add_library(strix_ds41_cuda STATIC ${STRIX_DS41_CUDA_SOURCES})
    target_link_libraries(strix_ds41_cuda PUBLIC strix_ds41_layer strix_ds41_formats strix_family_deepseek41 CUDA::cudart)
    target_include_directories(strix_ds41_cuda PUBLIC include)
    set(strix_ds41_cuda_targets strix_ds41_cuda)
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-core-tests tests/cuda/ds41_core_tests.cu)
        target_link_libraries(strix-cuda-ds41-core-tests PRIVATE strix_ds41_cuda strix_ds41_layer strix_ds41_formats)
        add_test(NAME strix-cuda-ds41-core-tests COMMAND strix-cuda-ds41-core-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-tests)
        # real weights (STRIX_DS41_FIXTURE_GGUF: dense tensors of layers 0-2, embedding, head); skips without it
        add_executable(strix-cuda-ds41-core-real-tests tests/cuda/ds41_core_real_tests.cu)
        target_link_libraries(strix-cuda-ds41-core-real-tests PRIVATE strix_ds41_cuda strix_family_deepseek41)
        add_test(NAME strix-cuda-ds41-core-real-tests COMMAND strix-cuda-ds41-core-real-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-real-tests)
        # timing of the layer core on real weights (not a test)
        add_executable(strix-cuda-ds41-core-bench tests/cuda/ds41_core_bench.cu)
        target_link_libraries(strix-cuda-ds41-core-bench PRIVATE strix_ds41_cuda strix_family_deepseek41)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-bench)
        # replay of a host-forward capture on the device (needs a capture and the weights; not a ctest)
        add_executable(strix-cuda-ds41-core-tap-check tests/cuda/ds41_core_tap_check.cu)
        target_link_libraries(strix-cuda-ds41-core-tap-check PRIVATE strix_ds41_cuda strix_family_deepseek41)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-tap-check)
        add_executable(strix-cuda-ds41-decode-fusions-tests tests/cuda/ds41_decode_fusions_tests.cu)
        target_link_libraries(strix-cuda-ds41-decode-fusions-tests PRIVATE strix_ds41_cuda)
        add_test(NAME strix-cuda-ds41-decode-fusions-tests COMMAND strix-cuda-ds41-decode-fusions-tests)
        set_tests_properties(strix-cuda-ds41-decode-fusions-tests PROPERTIES SKIP_RETURN_CODE 77)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-decode-fusions-tests)
    endif()
    # routed experts: decode GEMVs, prefill grouped GEMM, device routes and the combine (docs/models/deepseek-v4.1/cuda-numeric-identity.md)
    add_library(strix_ds41_cuda_moe STATIC src/cuda/ds41_moe.cu src/cuda/ds41_moe_grouped.cu src/cuda/ds41_moe_fast.cu src/cuda/ds41_moe_v2.cu)
    target_link_libraries(strix_ds41_cuda_moe PUBLIC strix_ds41_cuda strix_ds41_formats)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_moe)
    # v1 vs v2 decode kernels on real tensors (bitwise + interleaved timing)
    add_executable(strix-cuda-ds41-kern-bench tests/cuda/ds41_kern_bench.cu src/gguf_reader.cpp)
    target_link_libraries(strix-cuda-ds41-kern-bench PRIVATE strix_ds41_cuda_moe strix_ds41_cuda)
    set(strix_dsrtxk_bench_needs_parts_host ON)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-kern-bench)
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-moe-tests tests/cuda/ds41_moe_tests.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-tests PRIVATE strix_ds41_cuda_moe)
        add_test(NAME strix-cuda-ds41-moe-tests COMMAND strix-cuda-ds41-moe-tests)
        add_executable(strix-cuda-ds41-moe-grouped-tests tests/cuda/ds41_moe_grouped_tests.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-grouped-tests PRIVATE strix_ds41_cuda_moe)
        add_test(NAME strix-cuda-ds41-moe-grouped-tests COMMAND strix-cuda-ds41-moe-grouped-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-moe-grouped-tests)
        add_executable(strix-cuda-ds41-moe-bench tests/cuda/ds41_moe_bench.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-bench PRIVATE strix_ds41_cuda_moe)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-moe-tests strix-cuda-ds41-moe-bench)
        # the prefill fast arm against the exact path on a real layer (needs the experts GGUF; --bench: ms per layer)
        add_executable(strix-cuda-ds41-moe-fast-tests tests/cuda/ds41_moe_fast_tests.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-fast-tests PRIVATE strix_ds41_cuda_moe)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-moe-fast-tests)
    endif()
    add_library(strix_ds41_cuda_attn STATIC src/cuda/ds41_attention.cu src/cuda/ds41_attention_state.cu src/cuda/ds41_attention_export.cu
        src/cuda/ds41_fast_attention.cu src/cuda/ds41_fast_index.cu)
    target_link_libraries(strix_ds41_cuda_attn PUBLIC strix_ds41_cuda strix_ds41_attention strix_ds41_runtime)  # runtime: the decode state format
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_attn)
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-attention-tests tests/cuda/ds41_attention_tests.cu)
        target_link_libraries(strix-cuda-ds41-attention-tests PRIVATE strix_ds41_cuda_attn strix_ds41_layer strix_ds41_runtime)
        add_test(NAME strix-cuda-ds41-attention-tests COMMAND strix-cuda-ds41-attention-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-attention-tests)
        # timing of the attention ops at prefill shapes (not a test)
        add_executable(strix-cuda-ds41-attention-prefill-bench tests/cuda/ds41_attention_prefill_bench.cu)
        target_link_libraries(strix-cuda-ds41-attention-prefill-bench PRIVATE strix_ds41_cuda_attn)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-attention-prefill-bench)
        # fast prefill attention against the exact kernels (error of o, fused A8 bitwise) and the times of both
        add_executable(strix-cuda-ds41-fast-attention-tests tests/cuda/ds41_fast_attention_tests.cu)
        target_link_libraries(strix-cuda-ds41-fast-attention-tests PRIVATE strix_ds41_cuda_attn)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-fast-attention-tests)
    endif()
    # prefill driver: the DevicePrefill plan on the CUDA kernels, tiered routed experts, Engram rows from the file
    add_library(strix_ds41_cuda_prefill STATIC src/cuda/ds41_prefill_driver.cu src/cuda/ds41_prefill_kernels.cu src/cuda/ds41_prefill_q8.cu
        src/cuda/ds41_dense_fast.cu src/cuda/ds41_expert_streamer.cu src/cuda/ds41_gpu_profile.cu src/ds41_file_reader.cpp)
    target_link_libraries(strix_ds41_cuda_prefill PUBLIC strix_ds41_cuda_attn strix_ds41_cuda_moe strix_ds41_cuda strix_ds41_prefill
        strix_ds41_runtime strix_ds41_layer strix_family_deepseek41 ${CMAKE_DL_LIBS})
    if(STRIX_TIERSTREAM_DIR AND strix_mfile_define)
        target_link_libraries(strix_ds41_cuda_prefill PUBLIC strix_tierstream)
    endif()
    add_executable(strix-cuda-ds41-prefill apps/cuda_ds41_prefill_main.cu)
    target_link_libraries(strix-cuda-ds41-prefill PRIVATE strix_ds41_cuda_prefill)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_prefill strix-cuda-ds41-prefill)
    add_executable(strix-cuda-ds41-decode apps/cuda_ds41_decode_main.cu)
    target_link_libraries(strix-cuda-ds41-decode PRIVATE strix_ds41_cuda_prefill)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-decode)
    # pipeline FRONT on the CUDA decode step: the strix/ds41-pipeline-v1 messages (the portable part of the wire) and its
    # own client socket
    if(STRIX_CUDA_DS41_ONLY)
        add_library(strix_ds41_pipeline_messages STATIC src/ds41_pipeline_wire.cpp src/ds41_pipeline_multi.cpp src/ds41_pipeline_dspark.cpp)
        target_link_libraries(strix_ds41_pipeline_messages PUBLIC strix_family_deepseek41)
        set(strix_ds41_pipeline_for_front strix_ds41_pipeline_messages)
    else()
        set(strix_ds41_pipeline_for_front strix_ds41_pipeline)
    endif()
    add_executable(strix-cuda-ds41-pipeline-front apps/cuda_ds41_pipeline_front_main.cu)
    target_link_libraries(strix-cuda-ds41-pipeline-front PRIVATE strix_ds41_cuda_prefill ${strix_ds41_pipeline_for_front})
    if(WIN32)
        target_link_libraries(strix-cuda-ds41-pipeline-front PRIVATE ws2_32 winmm avrt)
    endif()
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-pipeline-front)
    # --profile default|fast|quality: the table-driven lever resolver (pure C++, CPU-tested; the replica and the service)
    add_library(strix_ds41_rtx_profile STATIC src/ds41_rtx_profile.cpp)
    target_include_directories(strix_ds41_rtx_profile PUBLIC include)
    target_compile_features(strix_ds41_rtx_profile PUBLIC cxx_std_20)
    # the prefill resident behind an HTTP/1.1 service (POST /v1/prefill, state handoff to a decode server's import route)
    add_executable(strix-cuda-ds41-prefill-service apps/cuda_ds41_prefill_service_main.cu src/serve/json.cpp)
    target_include_directories(strix-cuda-ds41-prefill-service PRIVATE src/serve)
    target_link_libraries(strix-cuda-ds41-prefill-service PRIVATE strix_ds41_cuda_prefill strix_core strix_ds41_rtx_profile)
    if(WIN32)
        target_link_libraries(strix-cuda-ds41-prefill-service PRIVATE ws2_32 bcrypt)
    endif()
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-prefill-service)
    # FRONT of several requests (strix/ds41-pipeline-multi-v1): sequence slots of one CudaPrefill, driven by a batch TAIL
    add_executable(strix-cuda-ds41-multi-front apps/cuda_ds41_multi_front_main.cu)
    target_link_libraries(strix-cuda-ds41-multi-front PRIVATE strix_ds41_cuda_prefill ${strix_ds41_pipeline_for_front})
    if(WIN32)
        target_link_libraries(strix-cuda-ds41-multi-front PRIVATE ws2_32 winmm avrt)
    endif()
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-multi-front)
    if(BUILD_TESTING)
        # prefill-shape lane-dot kernels bitwise against the decode kernels and the host references (--bench: times)
        add_executable(strix-cuda-ds41-prefill-kernels-tests tests/cuda/ds41_prefill_kernels_tests.cu)
        target_link_libraries(strix-cuda-ds41-prefill-kernels-tests PRIVATE strix_ds41_cuda_prefill)
        add_test(NAME strix-cuda-ds41-prefill-kernels-tests COMMAND strix-cuda-ds41-prefill-kernels-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-prefill-kernels-tests)
        # prefill-shape Q8_0 x A8 sites bitwise against the tiled kernel (--bench: times)
        add_executable(strix-cuda-ds41-prefill-q8-tests tests/cuda/ds41_prefill_q8_tests.cu)
        target_link_libraries(strix-cuda-ds41-prefill-q8-tests PRIVATE strix_ds41_cuda_prefill)
        add_test(NAME strix-cuda-ds41-prefill-q8-tests COMMAND strix-cuda-ds41-prefill-q8-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-prefill-q8-tests)
        # sustained RAM -> device bandwidth of the expert transfer shapes (not a test)
        add_executable(strix-cuda-ds41-xfer-bench tests/cuda/ds41_xfer_bench.cu)
        target_link_libraries(strix-cuda-ds41-xfer-bench PRIVATE strix_ds41_cuda)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-xfer-bench)
        # fast (relaxed FP32) dense sites against the exact kernels: errors per site (--bench: times)
        add_executable(strix-cuda-ds41-dense-fast-tests tests/cuda/ds41_dense_fast_tests.cu)
        target_link_libraries(strix-cuda-ds41-dense-fast-tests PRIVATE strix_ds41_cuda_prefill)
        add_test(NAME strix-cuda-ds41-dense-fast-tests COMMAND strix-cuda-ds41-dense-fast-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-dense-fast-tests)
    endif()
    # the Engram pages a prompt reads (to complete a partial model copy for that prompt)
    add_executable(strix-ds41-engram-pages apps/ds41_engram_pages_main.cpp)
    target_link_libraries(strix-ds41-engram-pages PRIVATE strix_ds41_layer strix_family_deepseek41)
    list(APPEND strix_ds41_cuda_targets strix-ds41-engram-pages)
    # parts-host: resident-expert routes, the seeded combine and the layer driver with expert servers
    add_library(strix_ds41_cuda_parts_host STATIC src/cuda/ds41_parts_host.cu src/cuda/ds41_parts_host_driver.cpp)
    target_link_libraries(strix_ds41_cuda_parts_host PUBLIC strix_ds41_cuda_moe strix_ds41_xrpc)
    if(strix_dsrtxk_bench_needs_parts_host)
        target_link_libraries(strix-cuda-ds41-kern-bench PRIVATE strix_ds41_cuda_parts_host)
    endif()
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_parts_host)
    # residency from a hot-expert placement (static + dynamic slots) and the decode replay over recorded routes
    add_library(strix_ds41_cuda_hot_residency STATIC src/cuda/ds41_hot_residency.cpp src/cuda/ds41_hot_site.cu src/gguf_reader.cpp)
    target_link_libraries(strix_ds41_cuda_hot_residency PUBLIC strix_ds41_cuda_parts_host strix_ds41_hot_placement CUDA::cuda_driver)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_hot_residency)
    add_executable(strix-cuda-ds41-parts-host-check apps/cuda_ds41_parts_host_check.cpp)
    target_link_libraries(strix-cuda-ds41-parts-host-check PRIVATE strix_ds41_cuda_hot_residency strix_ds41_cuda)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-parts-host-check)
    # decode with dense + hot experts: CudaPrefill::step with the parts-host at every MoE site
    add_executable(strix-cuda-ds41-hot-decode apps/cuda_ds41_hot_decode.cu)
    target_link_libraries(strix-cuda-ds41-hot-decode PRIVATE strix_ds41_cuda_hot_residency strix_ds41_cuda_prefill)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-hot-decode)
    # independent replica with streamed routed experts (design notes) on tierstream's device-gated decode mode
    # (device arena + pinned host tier + lookahead): needs a CUDA build of tierstream (STRIX_TIERSTREAM_DIR)
    if(STRIX_TIERSTREAM_DIR)
        target_link_libraries(strix_tierstream INTERFACE CUDA::cudart CUDA::cuda_driver)
        add_library(strix_ds41_cuda_tier STATIC src/cuda/ds41_tier_site.cu)
        target_link_libraries(strix_ds41_cuda_tier PUBLIC strix_ds41_cuda_hot_residency strix_tierstream)
        list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_tier)
        add_executable(strix-cuda-ds41-replica apps/cuda_ds41_replica.cu)
        target_link_libraries(strix-cuda-ds41-replica PRIVATE strix_ds41_cuda_tier strix_ds41_cuda_prefill strix_ds41_cuda_dspark
                              strix_ds41_rtx_profile)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${STRIX_TIERSTREAM_LIBRARY}")
        file(SHA256 "${STRIX_TIERSTREAM_LIBRARY}" strix_tierstream_sha256)
        target_compile_definitions(strix-cuda-ds41-replica PRIVATE STRIX_TIERSTREAM_SHA256="${strix_tierstream_sha256}")
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-replica)
        if(BUILD_TESTING)
            add_executable(strix-cuda-ds41-rtx-schedule-tests tests/ds41_rtx_schedule_tests.cpp)
            target_include_directories(strix-cuda-ds41-rtx-schedule-tests PRIVATE include)
            target_compile_features(strix-cuda-ds41-rtx-schedule-tests PRIVATE cxx_std_20)
            # whole-device budgets: tierstream's ss_budget_plan is pure CPU; linked from the tree's CPU build when it
            # has one (no CUDA runtime in a CPU test), else from the configured library
            set(strix_tierstream_cpu_lib "${STRIX_TIERSTREAM_DIR}/build-cpu/tierstream${CMAKE_STATIC_LIBRARY_SUFFIX}")
            if(EXISTS "${strix_tierstream_cpu_lib}")
                target_link_libraries(strix-cuda-ds41-rtx-schedule-tests PRIVATE "${strix_tierstream_cpu_lib}" Threads::Threads
                                      "$<$<PLATFORM_ID:Windows>:avrt;winmm;bcrypt>")
                target_include_directories(strix-cuda-ds41-rtx-schedule-tests PRIVATE "${STRIX_TIERSTREAM_DIR}/include")
            else()
                target_link_libraries(strix-cuda-ds41-rtx-schedule-tests PRIVATE strix_tierstream)
            endif()
            add_test(NAME strix-cuda-ds41-rtx-schedule-tests COMMAND strix-cuda-ds41-rtx-schedule-tests)
            set_tests_properties(strix-cuda-ds41-rtx-schedule-tests PROPERTIES LABELS "cpu;schedule")
            add_executable(strix-ds41-rtx-profile-tests tests/ds41_rtx_profile_tests.cpp)
            target_link_libraries(strix-ds41-rtx-profile-tests PRIVATE strix_ds41_rtx_profile)
            add_test(NAME strix-ds41-rtx-profile-tests COMMAND strix-ds41-rtx-profile-tests)
            set_tests_properties(strix-ds41-rtx-profile-tests PROPERTIES LABELS "cpu;profile")
            add_executable(strix-cuda-ds41-serve-tests tests/ds41_replica_serve_tests.cpp)
            target_compile_features(strix-cuda-ds41-serve-tests PRIVATE cxx_std_20)
            add_test(NAME strix-cuda-ds41-serve-tests COMMAND strix-cuda-ds41-serve-tests)
            set_tests_properties(strix-cuda-ds41-serve-tests PROPERTIES LABELS "cpu;serve")
            add_executable(strix-cuda-ds41-census-tests tests/ds41_replica_census_tests.cpp)
            target_include_directories(strix-cuda-ds41-census-tests PRIVATE include)
            target_compile_features(strix-cuda-ds41-census-tests PRIVATE cxx_std_20)
            add_test(NAME strix-cuda-ds41-census-tests COMMAND strix-cuda-ds41-census-tests)
            set_tests_properties(strix-cuda-ds41-census-tests PROPERTIES LABELS "cpu;census")
            add_executable(strix-ds41-rtx-drafter-select-tests tests/ds41_rtx_drafter_select_tests.cpp)
            target_include_directories(strix-ds41-rtx-drafter-select-tests PRIVATE include)
            target_compile_features(strix-ds41-rtx-drafter-select-tests PRIVATE cxx_std_20)
            add_test(NAME strix-ds41-rtx-drafter-select-tests COMMAND strix-ds41-rtx-drafter-select-tests)
            set_tests_properties(strix-ds41-rtx-drafter-select-tests PROPERTIES LABELS "cpu;drafter")
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
            add_custom_target(strix-cuda-ds41-config-tests DEPENDS strix-cuda-ds41-replica)
            add_test(NAME strix-cuda-ds41-config-tests
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/cuda_ds41_config_test.py
                    $<TARGET_FILE:strix-cuda-ds41-replica>)
            set_tests_properties(strix-cuda-ds41-config-tests PROPERTIES LABELS "cpu;config" TIMEOUT 60)
        endif()

    endif()
    # DSpark drafter on CUDA (design notes): the host reference library (built here in the CUDA-only tree) and
    # the device drafter, bitwise against the host reference on the real weights
    if(STRIX_CUDA_DS41_ONLY)
        add_library(strix_ds41_dspark STATIC src/ds41_dspark.cpp src/ds41_dspark_files.cpp src/ds41_dspark_driver.cpp src/ds41_dspark_vocab.cpp)
        target_link_libraries(strix_ds41_dspark PUBLIC strix_ds41_formats strix_core Threads::Threads)
        list(APPEND strix_ds41_cuda_targets strix_ds41_dspark)
        # the window controller's survival table + token identity under any table (CPU only)
        add_executable(strix-ds41-dspark-table-tests tests/ds41_dspark_table_tests.cpp)
        target_link_libraries(strix-ds41-dspark-table-tests PRIVATE strix_ds41_dspark)
        list(APPEND strix_ds41_cuda_targets strix-ds41-dspark-table-tests)
    endif()
    add_library(strix_ds41_cuda_dspark STATIC src/cuda/ds41_dspark.cu)
    target_link_libraries(strix_ds41_cuda_dspark PUBLIC strix_ds41_cuda_moe strix_ds41_cuda strix_ds41_dspark)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_dspark)
    add_executable(strix-cuda-ds41-dspark-tests tests/cuda/ds41_dspark_tests.cu)
    target_link_libraries(strix-cuda-ds41-dspark-tests PRIVATE strix_ds41_cuda_dspark)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-dspark-tests)
    add_executable(strix-cuda-ds41-hot-replay apps/cuda_ds41_hot_replay.cu)
    target_link_libraries(strix-cuda-ds41-hot-replay PRIVATE strix_ds41_cuda_hot_residency strix_ds41_cuda)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-hot-replay)
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-parts-host-tests tests/cuda/ds41_parts_host_tests.cu)
        target_link_libraries(strix-cuda-ds41-parts-host-tests PRIVATE strix_ds41_cuda_parts_host)
        add_test(NAME strix-cuda-ds41-parts-host-tests COMMAND strix-cuda-ds41-parts-host-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-parts-host-tests)
    endif()
    # Strict arithmetic: no contraction into FMA, IEEE division and square root, denormals kept.
    if(STRIX_ENABLE_CUDA_TRELLIS)
        add_library(strix_cuda_trellis STATIC src/cuda/trellis.cu)
        target_link_libraries(strix_cuda_trellis PUBLIC strix_ds41_cuda)
        list(APPEND strix_ds41_cuda_targets strix_cuda_trellis)
        # the decode step's trellis shared expert (CudaPrefillOptions::trellis_shexp)
        target_link_libraries(strix_ds41_cuda_prefill PUBLIC strix_cuda_trellis)
        target_compile_definitions(strix_ds41_cuda_prefill PRIVATE STRIX_DS41_CUDA_TRELLIS=1)
        if(BUILD_TESTING)
            add_executable(strix-cuda-trellis-tests tests/cuda/trellis_tests.cu)
            target_link_libraries(strix-cuda-trellis-tests PRIVATE strix_cuda_trellis)
            list(APPEND strix_ds41_cuda_targets strix-cuda-trellis-tests)
            # the decode step's trellis shared expert vs the pair's per-row outputs (a dump from the AMD node)
            add_executable(strix-cuda-trellis-shexp-dump-check tests/cuda/trellis_shexp_dump_check.cu)
            target_link_libraries(strix-cuda-trellis-shexp-dump-check PRIVATE strix_cuda_trellis)
            list(APPEND strix_ds41_cuda_targets strix-cuda-trellis-shexp-dump-check)
            add_test(NAME strix-cuda-trellis-contract-tests COMMAND strix-cuda-trellis-tests --contract-only)
            # Native runs require explicit external fixtures; do not silently
            # pass/skip a missing oracle or select a shared GPU during configure.
        endif()
    endif()
    foreach(target ${strix_ds41_cuda_targets})
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CXX_STANDARD 20)
        if(MSVC)
            # MSVC host compiler: /fp:precise (no contraction, see STRIX_CUDA_DS41_ONLY) in both the .cu host
            # passes and the plain C++ sources; the device flags are the same as on Linux.
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -prec-div=true -prec-sqrt=true -ftz=false
                    -Xcompiler=/fp:precise,/Zc:__cplusplus,/Zc:preprocessor,/permissive-,/utf-8,/bigobj,/EHsc>
                $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
        else()
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -prec-div=true -prec-sqrt=true -ftz=false -Xcompiler=-ffp-contract=off,-Wall,-Wextra>
                $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off -Wall -Wextra>)
        endif()
    endforeach()
endif()
endif()  # STRIX_FAMILY_DEEPSEEK41

# minimax-m3 bring-up: MSA and fused post-exchange host references, their gfx1151 kernels, and the layer gates.
if(STRIX_FAMILY_MINIMAX)
    # Defaults-on resolver (host only): the default-ON items, STRIX_MM_BASELINE=1 master switch, --print-effective-config.
    option(STRIX_MINIMAX_GU4_SCALAR "MiniMax gu4 MoE expert addressing (exact, 6a8d4ab6); OFF = the former reader" ON)
    if(STRIX_MINIMAX_GU4_SCALAR)
        set(strix_minimax_gu4 1)
    else()
        set(strix_minimax_gu4 0)
    endif()
    add_library(strix_minimax_defaults STATIC src/minimax/engine/mm_defaults.cpp)
    target_include_directories(strix_minimax_defaults PUBLIC src include)
    target_compile_features(strix_minimax_defaults PUBLIC cxx_std_20)
    target_compile_definitions(strix_minimax_defaults PRIVATE STRIX_MM_GU4_SCALAR=${strix_minimax_gu4})
    strix_target_warnings(strix_minimax_defaults)
    if(BUILD_TESTING)
        add_executable(strix-minimax-auto-tests tests/minimax_auto_tests.cpp src/serve/json.cpp)
        target_include_directories(strix-minimax-auto-tests PRIVATE src include)
        target_compile_features(strix-minimax-auto-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-minimax-auto-tests)
        add_test(NAME strix-minimax-auto-tests COMMAND strix-minimax-auto-tests)
        # Pure host policy census; source receipts reject unaudited selectors.
        add_executable(strix-minimax-route-identity-tests tests/minimax_route_identity_tests.cpp)
        target_link_libraries(strix-minimax-route-identity-tests PRIVATE strix_minimax_defaults)
        target_compile_definitions(strix-minimax-route-identity-tests PRIVATE STRIX_ROUTE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        strix_target_warnings(strix-minimax-route-identity-tests)
        if(NOT MSVC)
            target_compile_options(strix-minimax-route-identity-tests PRIVATE -Wsign-conversion)
        endif()
        add_test(NAME strix-minimax-route-identity-tests COMMAND strix-minimax-route-identity-tests)
        # Separate queue object: test clock/work counters never enter production or DeepSeek.
        add_executable(strix-minimax-lead-tests tests/minimax_lead_tests.cpp src/serve/engine.cpp src/serve/json.cpp
            src/minimax/dspark/minimax_dspark_host.cpp)
        target_compile_definitions(strix-minimax-lead-tests PRIVATE STRIX_SERVE_QUEUE_TEST=1)
        target_link_libraries(strix-minimax-lead-tests PRIVATE strix_lookup)
        target_include_directories(strix-minimax-lead-tests PRIVATE src include)
        target_compile_features(strix-minimax-lead-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-minimax-lead-tests)
        add_test(NAME strix-minimax-lead-tests COMMAND strix-minimax-lead-tests)
        add_executable(strix-minimax-sched-tests tests/minimax_sched_tests.cpp)
        target_link_libraries(strix-minimax-sched-tests PRIVATE strix_serve)
        target_include_directories(strix-minimax-sched-tests PRIVATE src include)
        strix_target_warnings(strix-minimax-sched-tests)
        add_test(NAME strix-minimax-sched-tests COMMAND strix-minimax-sched-tests)
        add_executable(strix-minimax-defaults-tests tests/minimax_defaults_tests.cpp)
        target_link_libraries(strix-minimax-defaults-tests PRIVATE strix_minimax_defaults)
        strix_target_warnings(strix-minimax-defaults-tests)
        add_test(NAME strix-minimax-defaults-tests COMMAND strix-minimax-defaults-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_production_receipt.txt)
    endif()
    add_library(strix_minimax_assets STATIC src/minimax/model_assets.cpp)
    target_include_directories(strix_minimax_assets PUBLIC include)
    target_link_libraries(strix_minimax_assets PUBLIC strix_serve)
    target_compile_features(strix_minimax_assets PUBLIC cxx_std_20)
    strix_target_warnings(strix_minimax_assets)

    add_library(strix_minimax_session STATIC src/minimax/session/session_cache.cpp)
    target_include_directories(strix_minimax_session PUBLIC include)
    target_link_libraries(strix_minimax_session PUBLIC strix_core strix_minimax_assets)
    target_compile_features(strix_minimax_session PUBLIC cxx_std_20)
    strix_target_warnings(strix_minimax_session)
    if(BUILD_TESTING)
        add_executable(strix-minimax-session-tests tests/minimax_session_cache_tests.cpp)
        target_link_libraries(strix-minimax-session-tests PRIVATE strix_minimax_session Threads::Threads)
        if(NOT WIN32)
            # Exercise the production pack reader with tiny host-only fixtures.
            target_sources(strix-minimax-session-tests PRIVATE src/minimax/engine/mm_pack.cpp)
            target_include_directories(strix-minimax-session-tests PRIVATE src)
        endif()
        strix_target_warnings(strix-minimax-session-tests)
        add_test(NAME strix-minimax-session-tests COMMAND strix-minimax-session-tests)
        add_executable(strix-minimax-session-api-tests tests/minimax_session_api_tests.cpp)
        target_link_libraries(strix-minimax-session-api-tests PRIVATE strix_serve Threads::Threads)
        target_include_directories(strix-minimax-session-api-tests PRIVATE src)
        strix_target_warnings(strix-minimax-session-api-tests)
        add_test(NAME strix-minimax-session-api-tests COMMAND strix-minimax-session-api-tests)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME minimax-session-gate-helpers COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_session_cache_gate_tests.py)
            set_tests_properties(minimax-session-gate-helpers PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
            add_test(NAME minimax-session-phasing COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_session_phasing_tests.py)
            add_test(NAME minimax-session-default-observer COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_session_default_probe_tests.py)
        endif()
    endif()
    # MiniMax tokenizer (tokenizer.json BPE: o200k-style split, NFC, byte level) and chat template, host only.
    add_library(strix_minimax_serve STATIC src/minimax/serve/minimax_tokenizer.cpp src/minimax/serve/minimax_chat.cpp
        src/minimax/serve/minimax_observe.cpp src/minimax/serve/minimax_api.cpp)
    target_include_directories(strix_minimax_serve PUBLIC src include)
    target_compile_features(strix_minimax_serve PUBLIC cxx_std_20)
    target_link_libraries(strix_minimax_serve PUBLIC strix_serve strix_lookup)
    strix_target_warnings(strix_minimax_serve)
    if(BUILD_TESTING)
        add_executable(strix-minimax-tokenizer-tests tests/minimax_tokenizer_tests.cpp)
        target_link_libraries(strix-minimax-tokenizer-tests PRIVATE strix_minimax_serve)
        strix_target_warnings(strix-minimax-tokenizer-tests)
        set(STRIX_MINIMAX_TOKENIZER_JSON "" CACHE FILEPATH "MiniMax tokenizer.json (optional; enables encode/decode checks)")
        add_test(NAME strix-minimax-tokenizer-tests COMMAND strix-minimax-tokenizer-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/src/minimax/serve/fixtures/minimax_tokenizer_fixtures.json ${STRIX_MINIMAX_TOKENIZER_JSON})
        # serve HTTP API and observability (request ids, status ring, metrics text, armed histograms), host only
        add_executable(strix-minimax-observe-tests tests/minimax_observe_tests.cpp)
        target_link_libraries(strix-minimax-observe-tests PRIVATE strix_minimax_serve)
        strix_target_warnings(strix-minimax-observe-tests)
        add_test(NAME strix-minimax-observe-tests COMMAND strix-minimax-observe-tests)
        add_executable(strix-minimax-api-tests tests/minimax_api_tests.cpp)
        target_link_libraries(strix-minimax-api-tests PRIVATE strix_minimax_serve)
        strix_target_warnings(strix-minimax-api-tests)
        add_test(NAME strix-minimax-api-tests COMMAND strix-minimax-api-tests)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME strix-minimax-serve-check-tests
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tools/minimax/serve_observe_check_test.py)
        endif()
    endif()
    add_library(strix_minimax_msa STATIC src/minimax_msa.cpp src/minimax_moe.cpp)
    target_include_directories(strix_minimax_msa PUBLIC include)
    target_compile_features(strix_minimax_msa PUBLIC cxx_std_20)
    strix_target_warnings(strix_minimax_msa)
    if(NOT MSVC)
        target_compile_options(strix_minimax_msa PRIVATE -ffp-contract=off)
    endif()
    if(BUILD_TESTING)
        add_executable(strix-minimax-msa-tests tests/minimax_msa_tests.cpp)
        target_link_libraries(strix-minimax-msa-tests PRIVATE strix_minimax_msa)
        strix_target_warnings(strix-minimax-msa-tests)
        if(NOT MSVC)
            target_compile_options(strix-minimax-msa-tests PRIVATE -ffp-contract=off)
        endif()
        add_test(NAME strix-minimax-msa-tests COMMAND strix-minimax-msa-tests)
    endif()
    if(STRIX_ENABLE_HIP_RUNTIME)
        add_library(strix_minimax_msa_hip STATIC src/hip_minimax_msa.hip src/hip_minimax_moe.hip)
        target_link_libraries(strix_minimax_msa_hip PUBLIC strix_minimax_msa "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix_minimax_msa_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
        set(strix_minimax_msa_hip_targets strix_minimax_msa_hip)
        # Optimized decode MoE / dense EXL3 kernels (src/minimax/moe/, minimax/moe-opt 92e664f8):
        # self-contained, no library link.
        add_library(strix_minimax_moe_opt_hip STATIC src/minimax/moe/mmx_moe.hip src/minimax/moe/mmx_prefill.hip
            src/minimax/rtxpf/fp32_identity.hip)
        # FP32 prefill identity: no contraction, the RTX PRO build must produce the same bits.
        set_source_files_properties(src/minimax/rtxpf/fp32_identity.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
        target_link_libraries(strix_minimax_moe_opt_hip PUBLIC strix_minimax_msa "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix_minimax_moe_opt_hip PUBLIC src include "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_definitions(strix_minimax_moe_opt_hip PRIVATE STRIX_MM_GU4_SCALAR=${strix_minimax_gu4})
        list(APPEND strix_minimax_msa_hip_targets strix_minimax_moe_opt_hip)
        if(BUILD_TESTING)
            add_executable(strix-hip-minimax-msa-tests tests/hip_minimax_msa_tests.hip)
            target_link_libraries(strix-hip-minimax-msa-tests PRIVATE strix_minimax_msa_hip)
            add_test(NAME strix-hip-minimax-msa-tests COMMAND strix-hip-minimax-msa-tests)
            list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-msa-tests)
            add_executable(strix-hip-minimax-moe-tests tests/hip_minimax_moe_tests.hip)
            target_link_libraries(strix-hip-minimax-moe-tests PRIVATE strix_minimax_msa_hip)
            add_test(NAME strix-hip-minimax-moe-tests COMMAND strix-hip-minimax-moe-tests)
            list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-moe-tests)
            add_executable(strix-hip-mm-fp32id-tests tests/hip_mm_fp32_identity_tests.hip)
            target_link_libraries(strix-hip-mm-fp32id-tests PRIVATE strix_minimax_moe_opt_hip)
            target_compile_options(strix-hip-mm-fp32id-tests PRIVATE -ffp-contract=off)
            add_test(NAME strix-hip-mm-fp32id-tests COMMAND strix-hip-mm-fp32id-tests)
            list(APPEND strix_minimax_msa_hip_targets strix-hip-mm-fp32id-tests)
        endif()
        # Layer gates G3 (one node) and G4 (two-rank TP): only with an external strix-trellis 0.3 tree (header + archive).
        set(STRIX_MINIMAX_TRELLIS_DIR ""
            CACHE PATH "strix-trellis 0.3 tree: include/strix_trellis.h and build-qualification-v03/libstrix_trellis.a")
        set(strix_minimax_trellis_lib "${STRIX_MINIMAX_TRELLIS_DIR}/build-qualification-v03/libstrix_trellis.a")
        if(BUILD_TESTING AND EXISTS "${STRIX_MINIMAX_TRELLIS_DIR}/include/strix_trellis.h" AND EXISTS "${strix_minimax_trellis_lib}")
            add_executable(strix-hip-minimax-g3 tests/hip_minimax_g3.hip)
            target_link_libraries(strix-hip-minimax-g3 PRIVATE strix_minimax_msa_hip "${strix_minimax_trellis_lib}")
            target_include_directories(strix-hip-minimax-g3 PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
            list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-g3)
            add_executable(strix-hip-mmx-moe-bench tests/hip_mmx_moe_bench.hip)
            target_link_libraries(strix-hip-mmx-moe-bench PRIVATE strix_minimax_moe_opt_hip strix_minimax_msa_hip "${strix_minimax_trellis_lib}")
            target_include_directories(strix-hip-mmx-moe-bench PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
            list(APPEND strix_minimax_msa_hip_targets strix-hip-mmx-moe-bench)
            if(TARGET strix_dist)
                add_executable(strix-hip-minimax-g4 tests/hip_minimax_g4.hip)
                target_link_libraries(strix-hip-minimax-g4 PRIVATE strix_minimax_msa_hip strix_dist "${strix_minimax_trellis_lib}")
                target_include_directories(strix-hip-minimax-g4 PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
                list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-g4)
            endif()
        endif()
        # MiniMax engine (fork of the DS41 TP2 engine): resident TP2 forward, gates, sessions, batched decode.
        if(TARGET strix_dist AND EXISTS "${STRIX_MINIMAX_TRELLIS_DIR}/include/strix_trellis.h" AND EXISTS "${strix_minimax_trellis_lib}")
            # Prefill exchange module (src/minimax/pfx): chunked wire, CRC lanes, in-place slots.
            add_library(strix_minimax_pfx STATIC src/minimax/pfx/pf_exchange.hip src/minimax/pfx/pf_crc.cpp)
            target_link_libraries(strix_minimax_pfx PUBLIC strix_dist "${STRIX_HIP_RUNTIME_LIBRARY}" Threads::Threads)
            target_include_directories(strix_minimax_pfx PUBLIC include src "${STRIX_HIP_INCLUDE_DIR}")
            set_target_properties(strix_minimax_pfx PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_features(strix_minimax_pfx PUBLIC cxx_std_20)
            target_compile_options(strix_minimax_pfx PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64>)
            strix_target_warnings(strix_minimax_pfx)
            if(BUILD_TESTING)
                add_executable(strix-hip-mm-pfx-tests tests/hip_mm_pfx_tests.hip)
                target_link_libraries(strix-hip-mm-pfx-tests PRIVATE strix_minimax_pfx)
                set_target_properties(strix-hip-mm-pfx-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
                target_compile_options(strix-hip-mm-pfx-tests PRIVATE -mno-wavefrontsize64)
            endif()
            add_library(strix_minimax_engine STATIC src/minimax/engine/engine.hip src/minimax/engine/draft_lin.hip src/minimax/engine/mm_pack.cpp
                src/minimax/attn/msa_fused.hip src/minimax/attn/index_proj.hip src/minimax/xfer/minimax_xfer.hip)
            target_link_libraries(strix_minimax_engine PUBLIC strix_minimax_assets strix_minimax_defaults strix_minimax_msa_hip strix_minimax_moe_opt_hip strix_minimax_pfx strix_dist "${strix_minimax_trellis_lib}")
            target_include_directories(strix_minimax_engine PUBLIC include src "${STRIX_HIP_INCLUDE_DIR}"
                PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
            set_target_properties(strix_minimax_engine PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_features(strix_minimax_engine PUBLIC cxx_std_20)
            target_compile_options(strix_minimax_engine PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64 -fno-fast-math
                -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee>)
            strix_target_warnings(strix_minimax_engine)
            # DSpark speculative draft (src/minimax/dspark): host reference + gfx1151 kernels.
            add_library(strix_minimax_dspark STATIC src/minimax/dspark/hip_minimax_dspark.hip src/minimax/dspark/markov_cand.hip src/minimax/dspark/minimax_dspark_host.cpp
                src/minimax/dspark/minimax_dspark_files.cpp src/minimax/dspark/minimax_ngram.cpp)
            target_include_directories(strix_minimax_dspark PUBLIC src/minimax/dspark "${STRIX_HIP_INCLUDE_DIR}")
            target_link_libraries(strix_minimax_dspark PUBLIC strix_minimax_assets "${STRIX_HIP_RUNTIME_LIBRARY}" Threads::Threads)
            set_target_properties(strix_minimax_dspark PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_features(strix_minimax_dspark PUBLIC cxx_std_20)
            target_compile_options(strix_minimax_dspark PRIVATE -fno-fast-math -ffp-contract=off $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64
                -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee>)
            strix_target_warnings(strix_minimax_dspark)
            if(BUILD_TESTING)
                add_executable(strix-mm-dspark-tests src/minimax/dspark/tests/minimax_dspark_tests.cpp)
                target_link_libraries(strix-mm-dspark-tests PRIVATE strix_minimax_dspark)
                target_compile_options(strix-mm-dspark-tests PRIVATE -ffp-contract=off)
                add_test(NAME strix-mm-dspark-tests COMMAND strix-mm-dspark-tests)
                add_executable(strix-hip-mm-markov-cand-tests src/minimax/dspark/tests/hip_markov_cand_tests.hip)
                target_link_libraries(strix-hip-mm-markov-cand-tests PRIVATE strix_minimax_dspark Threads::Threads)
                set_target_properties(strix-hip-mm-markov-cand-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
                target_compile_options(strix-hip-mm-markov-cand-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
                    -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
                add_executable(strix-hip-mm-dspark-tests src/minimax/dspark/tests/hip_minimax_dspark_tests.hip)
                target_link_libraries(strix-hip-mm-dspark-tests PRIVATE strix_minimax_dspark)
                set_target_properties(strix-hip-mm-dspark-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
                target_compile_options(strix-hip-mm-dspark-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
                    -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            endif()
            add_library(strix_minimax_session_runtime STATIC src/minimax/session/runtime.cpp)
            target_link_libraries(strix_minimax_session_runtime PUBLIC strix_minimax_session strix_minimax_engine strix_minimax_dspark)
            target_include_directories(strix_minimax_session_runtime PUBLIC src)
            target_compile_features(strix_minimax_session_runtime PUBLIC cxx_std_20)
            strix_target_warnings(strix_minimax_session_runtime)
            add_executable(strix-minimax-rank apps/minimax_rank_main.cpp src/bp_codec.cpp)  # bp-v1: RTX hand-off frames
            target_link_libraries(strix-minimax-rank PRIVATE strix_minimax_engine strix_minimax_dspark strix_minimax_serve strix_minimax_session_runtime)
            target_compile_features(strix-minimax-rank PRIVATE cxx_std_20)
            strix_target_warnings(strix-minimax-rank)
            if(BUILD_TESTING)  # drafter EXL3 linears: bitwise A/B + ABA timing
                add_executable(strix-hip-mmx-draft-bench tests/hip_mmx_draft_bench.hip)
                target_link_libraries(strix-hip-mmx-draft-bench PRIVATE strix_minimax_engine strix_minimax_dspark)
                target_include_directories(strix-hip-mmx-draft-bench PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
                set_target_properties(strix-hip-mmx-draft-bench PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
                target_compile_options(strix-hip-mmx-draft-bench PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off)
            endif()
            add_executable(strix-minimax-serve apps/minimax_serve_main.cpp)
            target_link_libraries(strix-minimax-serve PRIVATE strix_minimax_engine strix_minimax_serve strix_serve strix_minimax_dspark strix_minimax_session_runtime)
            target_compile_features(strix-minimax-serve PRIVATE cxx_std_20)
            strix_target_warnings(strix-minimax-serve)
        endif()
        foreach(target ${strix_minimax_msa_hip_targets})
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(${target} PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off
                -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(${target})
        endforeach()
    endif()
endif()
option(STRIX_FAMILY_H3 "Build the MiniMax H3 audio+video DiT family modules (opt-in)" OFF)
if(STRIX_FAMILY_H3)
    file(GLOB strix_h3_cmake "${CMAKE_CURRENT_SOURCE_DIR}/cmake/h3_*.cmake")
    list(SORT strix_h3_cmake)
    foreach(f IN LISTS strix_h3_cmake)
        include(${f})
    endforeach()
endif()
option(STRIX_FAMILY_GEMMA4 "Build the gemma4 (Nemotron 3.5 Bolt E4B) family modules" OFF)
if(STRIX_FAMILY_GEMMA4)
    file(GLOB strix_gemma4_cmake "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gemma4_*.cmake")
    foreach(f IN LISTS strix_gemma4_cmake)
        include(${f})
    endforeach()
endif()

# GLM-5.3-Flash engine (glm/engine): Tier A host emulation + CPU semantic test, Tier A / DSA device
# kernels + gate-K tests, and the TP2 engine (strix-glm5-rank) on strix-trellis (design notes).
option(STRIX_FAMILY_GLM5 "Build the GLM-5.3-Flash engine (glm/engine)" OFF)
if(STRIX_FAMILY_GLM5)
    add_library(strix_glm5_host STATIC src/glm/host/glm_host.cpp src/glm/spec/glm_spec_host.cpp src/glm/spec/glm_spec_merge.cpp)
    target_include_directories(strix_glm5_host PUBLIC src include)
    target_compile_features(strix_glm5_host PUBLIC cxx_std_20)
    target_compile_options(strix_glm5_host PRIVATE -ffp-contract=off -fno-fast-math)
    strix_target_warnings(strix_glm5_host)
    if(BUILD_TESTING)
        add_executable(strix-glm5-host-cpu-tests tests/glm_host_cpu_tests.cpp)
        target_link_libraries(strix-glm5-host-cpu-tests PRIVATE strix_glm5_host)
        target_compile_options(strix-glm5-host-cpu-tests PRIVATE -ffp-contract=off -fno-fast-math)
        add_executable(strix-glm5-spec-cpu-tests tests/glm_spec_cpu_tests.cpp)
        target_link_libraries(strix-glm5-spec-cpu-tests PRIVATE strix_glm5_host)
        target_compile_options(strix-glm5-spec-cpu-tests PRIVATE -ffp-contract=off -fno-fast-math)
        # load memory accounting (design notes): probe / admission / cost model / trace + watchdog / PackReader
        # streaming a synthetic multi-GiB pack (page cache and RSS bounded); host-only, Linux
        add_executable(strix-glm5-mem-cpu-tests tests/glm_mem_cpu_tests.cpp src/glm/engine/glm_mem.cpp src/glm/engine/glm_pack.cpp)
        target_include_directories(strix-glm5-mem-cpu-tests PRIVATE src include)
        target_compile_features(strix-glm5-mem-cpu-tests PRIVATE cxx_std_20)
        target_link_libraries(strix-glm5-mem-cpu-tests PRIVATE Threads::Threads)
        strix_target_warnings(strix-glm5-mem-cpu-tests)
        # strix-glm5-rank defaults: plain invocation == production receipt, off switches / baseline == old path (host-only)
        add_executable(strix-glm5-rank-defaults-cpu-tests tests/glm_rank_defaults_cpu_tests.cpp src/glm/engine/glm_mem.cpp)
        target_link_libraries(strix-glm5-rank-defaults-cpu-tests PRIVATE strix_glm5_host Threads::Threads)
        target_include_directories(strix-glm5-rank-defaults-cpu-tests PRIVATE apps)
        target_compile_features(strix-glm5-rank-defaults-cpu-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-glm5-rank-defaults-cpu-tests)
        add_test(NAME strix-glm5-rank-defaults-cpu-tests COMMAND strix-glm5-rank-defaults-cpu-tests)
        if(TARGET strix_serve)
            # strix-glm5-serve control channel, lead / follower protocol and scheduling on a fake engine (host-only)
            add_executable(strix-glm5-serve-cpu-tests tests/glm_serve_cpu_tests.cpp)
            target_link_libraries(strix-glm5-serve-cpu-tests PRIVATE strix_serve strix_glm5_host Threads::Threads)
            target_include_directories(strix-glm5-serve-cpu-tests PRIVATE src include)
            target_compile_features(strix-glm5-serve-cpu-tests PRIVATE cxx_std_20)
            strix_target_warnings(strix-glm5-serve-cpu-tests)
            add_test(NAME strix-glm5-serve-cpu-tests COMMAND strix-glm5-serve-cpu-tests)
        endif()
    endif()
    if(STRIX_ENABLE_HIP_RUNTIME)
        set(strix_glm5_hip_flags -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt
            -fdenormal-fp-math=ieee -fno-gpu-flush-denormals-to-zero)
        add_library(strix_glm5_kernels STATIC src/glm/kernels/glm_tier_a.hip src/glm/kernels/glm_dsa.hip src/glm/attn/glm_attn_fast.hip src/glm/spec/glm_spec_dev.hip
            src/glm/prefill/glm_prefill.hip src/glm/xfer/glm_post_fast.hip src/glm/moe/glm_moe.hip src/glm/engine/glm_r1b.hip src/glm/attn/glm_attn_tierb.hip)
        target_link_libraries(strix_glm5_kernels PUBLIC strix_glm5_host "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix_glm5_kernels PUBLIC src include "${STRIX_HIP_INCLUDE_DIR}")
        set(strix_glm5_hip_targets strix_glm5_kernels)
        if(BUILD_TESTING)
            add_executable(strix-hip-glm5-tier-a-tests tests/hip_glm_tier_a_tests.hip)
            target_link_libraries(strix-hip-glm5-tier-a-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-dsa-tests tests/hip_glm_dsa_tests.hip)
            target_link_libraries(strix-hip-glm5-dsa-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-attn-fast-tests tests/hip_glm_attn_fast_tests.hip)
            target_link_libraries(strix-hip-glm5-attn-fast-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-spec-tests tests/hip_glm_spec_tests.hip)
            target_link_libraries(strix-hip-glm5-spec-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-r1b-tests tests/hip_glm_r1b_tests.hip)
            target_link_libraries(strix-hip-glm5-r1b-tests PRIVATE strix_glm5_kernels)
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-r1b-tests)
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-tier-a-tests strix-hip-glm5-dsa-tests strix-hip-glm5-attn-fast-tests
                 strix-hip-glm5-spec-tests)
        endif()
        set(STRIX_GLM5_TRELLIS_DIR ""
            CACHE PATH "strix-trellis tree linked by the GLM engine (Tier B reference): include/strix_trellis.h + build-qualification-v03/libstrix_trellis.a")
        set(strix_glm5_trellis_lib "${STRIX_GLM5_TRELLIS_DIR}/build-qualification-v03/libstrix_trellis.a")
        if(TARGET strix_dist AND EXISTS "${STRIX_GLM5_TRELLIS_DIR}/include/strix_trellis.h" AND EXISTS "${strix_glm5_trellis_lib}")
            add_library(strix_glm5_engine STATIC src/glm/engine/engine.hip src/glm/engine/glm_fast_attn.hip src/glm/engine/glm_fast_moe.hip
                src/glm/engine/glm_pack.cpp src/glm/engine/glm_mem.cpp
                src/glm/xfer/glm_xfer.hip src/glm/xfer/glm_xfer_exchange.hip src/glm/xfer/glm_post_exchange.hip
                src/glm/prefill/glm_pf_moe.hip src/glm/prefill/glm_pf_path.hip src/glm/prefill/glm_pf_exact.hip src/glm/prefill/glm_pf_exact_moe.hip)
            target_link_libraries(strix_glm5_engine PUBLIC strix_glm5_kernels strix_dist "${strix_glm5_trellis_lib}")
            target_include_directories(strix_glm5_engine PUBLIC include src "${STRIX_HIP_INCLUDE_DIR}"
                PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix_glm5_engine)
            if(BUILD_TESTING)
                add_executable(strix-hip-glm5-post-exchange-rows-tests tests/hip_glm_post_exchange_rows_tests.hip)
                target_link_libraries(strix-hip-glm5-post-exchange-rows-tests PRIVATE strix_glm5_engine)
                list(APPEND strix_glm5_hip_targets strix-hip-glm5-post-exchange-rows-tests)
                # FastAttn multi-slot verify windows == single-slot verify calls, bitwise
                add_executable(strix-hip-glm5-fast-verify-tests tests/hip_glm_fast_verify_tests.hip)
                target_link_libraries(strix-hip-glm5-fast-verify-tests PRIVATE strix_glm5_engine)
                list(APPEND strix_glm5_hip_targets strix-hip-glm5-fast-verify-tests)
            endif()
            add_executable(strix-glm5-g3-replay tests/glm_g3_replay.hip)
            target_link_libraries(strix-glm5-g3-replay PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-g3-replay PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-glm5-g3-replay)
            # fused attention Tier B: gate (bitwise vs the linked strix_gemv) + bench
            add_executable(strix-hip-glm5-attn-tierb-tests tests/hip_glm_attn_tierb_tests.hip)
            target_link_libraries(strix-hip-glm5-attn-tierb-tests PRIVATE strix_glm5_kernels "${strix_glm5_trellis_lib}")
            target_include_directories(strix-hip-glm5-attn-tierb-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-attn-tierb-tests)
            add_executable(strix-glm5-gate-d-digest tests/glm_gate_d_digest.hip)
            target_link_libraries(strix-glm5-gate-d-digest PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-gate-d-digest PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_include_directories(strix-glm5-gate-d-digest SYSTEM PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/src")
            list(APPEND strix_glm5_hip_targets strix-glm5-gate-d-digest)
            add_executable(strix-glm5-decode-q tools/glm/omath/decode_q.hip)
            target_link_libraries(strix-glm5-decode-q PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-decode-q PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_include_directories(strix-glm5-decode-q SYSTEM PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/src")
            list(APPEND strix_glm5_hip_targets strix-glm5-decode-q)
            # chunked prefill gate / timing app and the prefill gates (design notes)
            add_executable(strix-glm5-pf apps/glm_pf_main.cpp)
            target_link_libraries(strix-glm5-pf PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-pf PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_compile_features(strix-glm5-pf PRIVATE cxx_std_20)
            strix_target_warnings(strix-glm5-pf)
            add_executable(strix-hip-glm5-prefill-tests tests/hip_glm_prefill_tests.hip)
            target_link_libraries(strix-hip-glm5-prefill-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-pf-tierb-tests tests/hip_glm_pf_tierb_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-tierb-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-tierb-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            add_executable(strix-hip-glm5-pf-exact-tests tests/hip_glm_pf_exact_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-exact-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-exact-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-exact-tests)
            add_executable(strix-hip-glm5-pf-layer-bench tests/hip_glm_pf_layer_bench.hip)  # MLA / layer stage timing (no gate)
            target_link_libraries(strix-hip-glm5-pf-layer-bench PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-layer-bench PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            add_executable(strix-hip-glm5-pf-exact-moe-tests tests/hip_glm_pf_exact_moe_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-exact-moe-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-exact-moe-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-exact-moe-tests)
            add_executable(strix-hip-glm5-pf-moe-tests tests/hip_glm_pf_moe_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-moe-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-moe-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-prefill-tests strix-hip-glm5-pf-tierb-tests strix-hip-glm5-pf-moe-tests)
            add_executable(strix-glm5-rank apps/glm_rank_main.cpp src/glm/spec/glm_spec_run.cpp)
            target_link_libraries(strix-glm5-rank PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-rank PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_compile_features(strix-glm5-rank PRIVATE cxx_std_20)
            strix_target_warnings(strix-glm5-rank)
            if(TARGET strix_serve)
                # TP2 HTTP server with continuous batched decode (rank 0 lead + HTTP, rank 1 follower)
                add_executable(strix-glm5-serve apps/glm_serve_main.cpp src/glm/spec/glm_spec_run.cpp)
                target_link_libraries(strix-glm5-serve PRIVATE strix_glm5_engine strix_serve)
                target_include_directories(strix-glm5-serve PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
                target_compile_features(strix-glm5-serve PRIVATE cxx_std_20)
                strix_target_warnings(strix-glm5-serve)
            endif()
        endif()
        foreach(target ${strix_glm5_hip_targets})
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${strix_glm5_hip_flags}>)
            strix_target_warnings(${target})
        endforeach()
        foreach(target strix-hip-glm5-prefill-tests strix-hip-glm5-pf-tierb-tests strix-hip-glm5-pf-moe-tests strix-hip-glm5-pf-exact-tests strix-hip-glm5-pf-exact-moe-tests)  # prefill gates
            if(TARGET ${target})
                target_compile_options(${target} PRIVATE -Wno-sign-conversion)
            endif()
        endforeach()
        if(TARGET strix-hip-glm5-attn-fast-tests)  # attention gate compiled without -Wsign-conversion
            target_compile_options(strix-hip-glm5-attn-fast-tests PRIVATE -Wno-sign-conversion)
        endif()
    endif()
endif()

# Host-only RTX session envelope/LRU tests (also available in CUDA-only builds).
if(BUILD_TESTING AND STRIX_FAMILY_DEEPSEEK41)
    add_executable(strix-ds41-session-cache-tests tests/ds41_session_cache_test.cpp src/deepseek41_attention_state.cpp)
    target_link_libraries(strix-ds41-session-cache-tests PRIVATE strix_core strix_ds41_attention strix_ds41_formats Threads::Threads)
    target_include_directories(strix-ds41-session-cache-tests PRIVATE include)
    target_compile_features(strix-ds41-session-cache-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_definitions(strix-ds41-session-cache-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-ds41-session-cache-tests)
    add_test(NAME strix-ds41-session-cache-tests COMMAND strix-ds41-session-cache-tests
        "${CMAKE_CURRENT_BINARY_DIR}/session-test-${CMAKE_BUILD_TYPE}")
    add_executable(strix-ds41-session-admission-tests tests/ds41_session_admission_test.cpp)
    target_link_libraries(strix-ds41-session-admission-tests PRIVATE strix_core)
    target_include_directories(strix-ds41-session-admission-tests PRIVATE include)
    target_compile_features(strix-ds41-session-admission-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-ds41-session-admission-tests)
    add_test(NAME strix-ds41-session-admission-tests COMMAND strix-ds41-session-admission-tests
        "${CMAKE_CURRENT_BINARY_DIR}/session-admission-test-${CMAKE_BUILD_TYPE}")
    set_tests_properties(strix-ds41-session-admission-tests PROPERTIES LABELS "cpu;session" TIMEOUT 120)
endif()

# Full-content hash checks are also available in the CUDA-only host library.
if(BUILD_TESTING AND STRIX_CUDA_DS41_ONLY)
    add_executable(strix-model-content-hash-tests tests/model_content_hash_tests.cpp)
    target_link_libraries(strix-model-content-hash-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-content-hash-tests)
    add_test(NAME strix-model-content-hash-tests COMMAND strix-model-content-hash-tests)
endif()
if(BUILD_TESTING AND STRIX_FAMILY_DEEPSEEK41)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME strix-rtx-session-gate-tests
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rtx_session_gate.py)
    set_tests_properties(strix-ds41-session-cache-tests strix-rtx-session-gate-tests PROPERTIES LABELS "cpu;session" TIMEOUT 120)
endif()
