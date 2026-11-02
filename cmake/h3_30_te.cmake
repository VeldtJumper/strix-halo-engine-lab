# MiniMax H3 text encoder (K8): host tokenizer + presentation; prefill planned on the dense path.
add_library(strix_h3_te STATIC
    src/h3/te/te_tokenizer.cpp
    src/h3/te/te_vision_host.cpp
    src/serve/json.cpp)
target_include_directories(strix_h3_te PUBLIC src include)
target_link_libraries(strix_h3_te PUBLIC strix_h3_common)
target_compile_features(strix_h3_te PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(strix_h3_te PRIVATE /utf-8)
    target_compile_definitions(strix_h3_te PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(BUILD_TESTING)
    # Directory with vocab.json, merges.txt, tokenizer_config.json of the Qwen2/3 tokenizer.
    set(STRIX_H3_TE_TOKENIZER_DIR "" CACHE PATH "Qwen tokenizer files for the H3 text-encoder tokenizer tests")
    add_executable(strix-h3-te-tests tests/h3/te/te_tokenizer_tests.cpp)
    target_link_libraries(strix-h3-te-tests PRIVATE strix_h3_te)
    if(MSVC)
        target_compile_options(strix-h3-te-tests PRIVATE /utf-8)
        target_compile_definitions(strix-h3-te-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME h3-te-host COMMAND strix-h3-te-tests host)
    add_test(NAME h3-te-tokenizer COMMAND strix-h3-te-tests tokenizer
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/h3/te/tok_cases.json" "${STRIX_H3_TE_TOKENIZER_DIR}")
    set(STRIX_H3_TE_VISION_FIXTURES "" CACHE PATH "H3 text-encoder vision fixture dir (manifest.json + <case>.*.f32)")
    add_test(NAME h3-te-vision COMMAND strix-h3-te-tests vision "${STRIX_H3_TE_VISION_FIXTURES}")
    set_tests_properties(h3-te-host h3-te-tokenizer h3-te-vision PROPERTIES TIMEOUT 120)
endif()

# GPU text path (gfx1151). The pack reader is host code; the GPU gate is a separate executable.
add_library(strix_h3_te_pack STATIC src/h3/te/te_pack_reader.cpp src/h3/te/te_sha256.cpp)
target_link_libraries(strix_h3_te_pack PUBLIC strix_h3_te)
if(MSVC)
    target_compile_options(strix_h3_te_pack PRIVATE /utf-8)
    target_compile_definitions(strix_h3_te_pack PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(BUILD_TESTING)
    add_executable(strix-h3-te-pack-tests tests/h3/te/te_pack_tests.cpp)
    target_link_libraries(strix-h3-te-pack-tests PRIVATE strix_h3_te_pack)
    add_test(NAME h3-te-pack COMMAND strix-h3-te-pack-tests "${CMAKE_CURRENT_BINARY_DIR}")
    set_tests_properties(h3-te-pack PROPERTIES TIMEOUT 60)
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_h3_te_gpu STATIC src/h3/te/te_gpu.hip)
    set_target_properties(strix_h3_te_gpu PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHITECTURES}")
    target_compile_options(strix_h3_te_gpu PRIVATE
        -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
    target_link_libraries(strix_h3_te_gpu PUBLIC strix_h3_te_pack strix_hip_runtime)
    if(BUILD_TESTING)
        # GPU gate; needs the pack and the fixtures, so it is not registered with CTest.
        add_executable(strix-h3-te-gpu-tests tests/h3/te/te_gpu_tests.cpp)
        target_link_libraries(strix-h3-te-gpu-tests PRIVATE strix_h3_te_gpu)
    endif()
endif()
