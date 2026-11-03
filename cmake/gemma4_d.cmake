add_library(strix_gemma4_serve STATIC
    src/gemma4/serve/tokenizer.cpp
    src/gemma4/serve/chat.cpp
    src/gemma4/serve/parse.cpp
    src/gemma4/serve/adapter.cpp)
target_include_directories(strix_gemma4_serve PUBLIC src include)
target_compile_features(strix_gemma4_serve PUBLIC cxx_std_20)
if(WIN32)
    # The shared HTTP transport is POSIX-only. Reuse its portable sources unchanged
    # for host qualification; Linux product builds link the complete shared stack.
    add_library(strix_gemma4_serve_support STATIC src/serve/json.cpp
        src/serve/bpe_tokenizer.cpp src/serve/chat_template.cpp)
    target_include_directories(strix_gemma4_serve_support PUBLIC include)
    target_compile_features(strix_gemma4_serve_support PUBLIC cxx_std_20)
    target_compile_options(strix_gemma4_serve_support PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    target_link_libraries(strix_gemma4_serve PUBLIC strix_gemma4_serve_support)
else()
    target_link_libraries(strix_gemma4_serve PUBLIC strix_serve)
endif()
strix_target_warnings(strix_gemma4_serve)
if(MSVC)
    target_compile_options(strix_gemma4_serve PRIVATE /utf-8)
    target_compile_definitions(strix_gemma4_serve PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(BUILD_TESTING)
    set(STRIX_GEMMA4_TOKENIZER_JSON "" CACHE FILEPATH "Official Gemma 4 tokenizer.json for asset qualification")
    add_executable(strix-gemma4-serve-tests tests/gemma4_serve/serve_tests.cpp)
    target_link_libraries(strix-gemma4-serve-tests PRIVATE strix_gemma4_serve)
    strix_target_warnings(strix-gemma4-serve-tests)
    if(MSVC)
        target_compile_options(strix-gemma4-serve-tests PRIVATE /utf-8)
    endif()
    add_test(NAME gemma4-serve-host COMMAND strix-gemma4-serve-tests host
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_serve/fixtures")
    add_test(NAME gemma4-serve-tokenizer COMMAND strix-gemma4-serve-tests tokenizer
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_serve/fixtures" "${STRIX_GEMMA4_TOKENIZER_JSON}")
    add_test(NAME gemma4-serve-tokenizer-reference COMMAND strix-gemma4-serve-tests tokenizer
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_serve/fixtures" "${STRIX_GEMMA4_TOKENIZER_JSON}")
    set_tests_properties(gemma4-serve-tokenizer-reference PROPERTIES ENVIRONMENT "STRIX_G4_TOKENIZER_HEAP=0")
    set_tests_properties(gemma4-serve-host gemma4-serve-tokenizer gemma4-serve-tokenizer-reference PROPERTIES TIMEOUT 60)
    set_tests_properties(gemma4-serve-tokenizer gemma4-serve-tokenizer-reference PROPERTIES SKIP_RETURN_CODE 77)
endif()
# strix-gemma4-tokenize: text <-> IDs with the engine's own tokenizer.json path (host only).
add_executable(strix-gemma4-tokenize apps/gemma4_tokenize_main.cpp)
target_link_libraries(strix-gemma4-tokenize PRIVATE strix_gemma4_serve)
strix_target_warnings(strix-gemma4-tokenize)
if(MSVC)
    target_compile_options(strix-gemma4-tokenize PRIVATE /utf-8)
endif()
if(BUILD_TESTING)
    add_executable(strix-gemma4-tokenize-cli-tests tests/gemma4_serve/tokenize_cli_tests.cpp)
    target_link_libraries(strix-gemma4-tokenize-cli-tests PRIVATE strix_gemma4_serve)
    strix_target_warnings(strix-gemma4-tokenize-cli-tests)
    if(MSVC)
        target_compile_options(strix-gemma4-tokenize-cli-tests PRIVATE /utf-8)
    endif()
    add_test(NAME gemma4-tokenize-cli COMMAND strix-gemma4-tokenize-cli-tests $<TARGET_FILE:strix-gemma4-tokenize>
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_serve/fixtures" "${CMAKE_CURRENT_BINARY_DIR}/gemma4-tokenize-cli"
        "${STRIX_GEMMA4_TOKENIZER_JSON}")
    set_tests_properties(gemma4-tokenize-cli PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
endif()
