cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 120)
project(glm_native_tools LANGUAGES CXX CUDA)
# Standalone configure. No ccache, device execution or compiler-generated native
# architecture probe. The only download is the pinned CUTLASS release below,
# skipped when GLM_CUTLASS_DIR names an existing CUTLASS 4.2.1 checkout (offline).
get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
enable_testing()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/O2> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
  $<$<COMPILE_LANGUAGE:CXX>:/fp:strict> $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
  $<$<COMPILE_LANGUAGE:CUDA>:-ftz=false> $<$<COMPILE_LANGUAGE:CUDA>:-prec-div=true>
  $<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=true> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:strict,/EHsc,/O2,/MD,/arch:AVX2>)
# CUTLASS 4.2.1 (BSD-3-Clause, header-only use): pinned release tarball with its
# SHA-256; GLM_CUTLASS_DIR overrides it with a local checkout of the same tag.
set(GLM_CUTLASS_DIR "" CACHE PATH "existing CUTLASS v4.2.1 source tree (offline builds); empty = fetch the pinned release")
if(GLM_CUTLASS_DIR)
  set(GLM_CUTLASS_SOURCE "${GLM_CUTLASS_DIR}")
else()
  include(FetchContent)
  FetchContent_Declare(glm_cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.2.1.tar.gz
    URL_HASH SHA256=a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR glm-headers-only) # no CMakeLists there: headers only, CUTLASS's own build is not configured
  FetchContent_MakeAvailable(glm_cutlass)
  set(GLM_CUTLASS_SOURCE "${glm_cutlass_SOURCE_DIR}")
endif()
if(NOT EXISTS "${GLM_CUTLASS_SOURCE}/include/cutlass/version.h")
  message(FATAL_ERROR "CUTLASS headers not found under ${GLM_CUTLASS_SOURCE}/include")
endif()
file(STRINGS "${GLM_CUTLASS_SOURCE}/include/cutlass/version.h" glm_cutlass_version REGEX "^#define CUTLASS_(MAJOR|MINOR|PATCH) ")
if(NOT glm_cutlass_version MATCHES "MAJOR 4;.*MINOR 2;.*PATCH 1")
  message(FATAL_ERROR "CUTLASS 4.2.1 required, ${GLM_CUTLASS_SOURCE} has: ${glm_cutlass_version}")
endif()
include_directories("${REPO}/include" "${REPO}/src" "${GLM_CUTLASS_SOURCE}/include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
add_library(glm_host_hash STATIC "${REPO}/src/model_content_hash.cpp" "${REPO}/src/model_runtime_binding.cpp"
  "${REPO}/src/model_manifest.cpp" "${REPO}/src/gguf_reader.cpp")
add_library(glm_json STATIC "${REPO}/src/serve/json.cpp")
# --profile default|fast|quality resolver (pure C++: the service and the CPU host test).
add_library(glm_rtx_profile STATIC profile.cpp)
target_link_libraries(glm_host_hash PUBLIC crypt32 bcrypt advapi32)
add_executable(glm_native_integrity integrity.cpp)
target_link_libraries(glm_native_integrity PRIVATE glm_host_hash glm_json)
foreach(kind receiver sender shards policy verified_io source_inventory decode_host)
  add_executable(glm_native_${kind} "${REPO}/tests/glm_native/${kind}.cpp")
  target_link_libraries(glm_native_${kind} PRIVATE glm_host_hash glm_json ws2_32)
endforeach()
target_link_libraries(glm_native_decode_host PRIVATE glm_rtx_profile)
add_test(NAME producer_policy COMMAND glm_native_policy)
add_test(NAME producer_verified_io COMMAND glm_native_verified_io "${CMAKE_BINARY_DIR}")
add_test(NAME producer_receiver_host COMMAND glm_native_receiver)
add_test(NAME producer_shard_sources COMMAND python -B "${REPO}/tests/glm_native/test_shards.py" --build "${CMAKE_BINARY_DIR}")
# The tierstream descriptor test compiles the real desc-v1 parser from a
# read-only tierstream source tree; absent -> the test reports SKIP for that part.
set(GLM_TIERSTREAM_SRC "${REPO}/../tierstream-d2569df" CACHE PATH "private tierstream snapshot (git archive of tierstream d2569df + STAMP): cache library, budget planner/ledger, desc parser")
# The tierstream commit this producer is qualified against (bound opens with
# expected source identities, ss_budget_plan / ss_budget_ledger). The source is a
# `git archive` export of that commit (not a moving checkout); its STAMP
# file names the exported commit and must match the pin, or configuration fails.
# GLM_TIERSTREAM_QUALIFIED_PIN is not a cache entry: an existing build tree keeps its cached
# GLM_TIERSTREAM_PIN/SRC across a pin bump, so configuration warns whenever they differ from it.
set(GLM_TIERSTREAM_QUALIFIED_PIN "d2569df")
set(GLM_TIERSTREAM_PIN "${GLM_TIERSTREAM_QUALIFIED_PIN}" CACHE STRING "required tierstream commit (prefix)")
if(NOT GLM_TIERSTREAM_PIN STREQUAL GLM_TIERSTREAM_QUALIFIED_PIN)
  message(WARNING "GLM_TIERSTREAM_PIN is ${GLM_TIERSTREAM_PIN}, not the qualified ${GLM_TIERSTREAM_QUALIFIED_PIN}: an override (A/B) or a stale cache from an older tree - reconfigure with -DGLM_TIERSTREAM_PIN=${GLM_TIERSTREAM_QUALIFIED_PIN} -DGLM_TIERSTREAM_SRC=<its snapshot> (or a fresh build) for a qualified binary")
endif()
# The host test publishes the service-shaped receipt arm (fresh each run) that the
# Python test reads: an explicit fixture pair, never a stale or racing artifact.
set(GLM_DECODE_RECEIPT "${CMAKE_BINARY_DIR}/ctest-fixtures/decode-receipt-arm.json")
add_test(NAME producer_decode_host COMMAND glm_native_decode_host "${CMAKE_BINARY_DIR}" "${GLM_DECODE_RECEIPT}")
set_tests_properties(producer_decode_host PROPERTIES FIXTURES_SETUP glm_decode_receipt)
add_test(NAME producer_decode_python COMMAND python -B "${REPO}/tests/glm_native/test_decode.py" --build "${CMAKE_BINARY_DIR}" --receipt "${GLM_DECODE_RECEIPT}")
set_tests_properties(producer_decode_python PROPERTIES FIXTURES_REQUIRED glm_decode_receipt)
foreach(kind native kda dsa head)
  add_library(producer_${kind} STATIC "../${kind}.cu")
endforeach()
add_library(producer_runtime STATIC runtime.cu)
add_executable(glm_native_service service.cpp)
target_link_libraries(glm_native_service PRIVATE glm_rtx_profile producer_runtime producer_head producer_dsa producer_kda producer_native glm_json glm_host_hash ws2_32 cuda cudart_static user32 advapi32)
target_link_directories(glm_native_service PRIVATE "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/lib/x64")
set(BENCH "${CMAKE_CURRENT_LIST_DIR}/../bench")
foreach(kind host kda_host dsa_host)
  add_executable(${kind}_tests "${BENCH}/${kind}_tests.cpp")
  add_test(NAME ${kind}_tests COMMAND ${kind}_tests)
endforeach()
add_executable(glm_native_bench "${BENCH}/bench.cu")
target_link_libraries(glm_native_bench PRIVATE producer_native)
add_executable(glm_kda_bench "${BENCH}/kda_bench.cu")
target_link_libraries(glm_kda_bench PRIVATE producer_kda)
add_executable(glm_dsa_bench "${BENCH}/dsa_bench.cu")
target_link_libraries(glm_dsa_bench PRIVATE producer_dsa producer_native)
# GPU exactness of the few-row trellis kernels (row_mma.cuh) vs the CUTLASS paths; run by hand on a GPU.
add_executable(glm_row_mma_test "${BENCH}/row_mma_test.cu")
target_link_libraries(glm_row_mma_test PRIVATE producer_native)
# The shared tierstream cache library built from its read-only source into
# this build directory: a CUDA build for the service, a CPU-only build for the
# CPU admission tests (no test process loads the CUDA driver). The archive is
# linked against this project's cudart_static: one CUDA runtime per process.
set(GLM_TS_HAVE OFF)
if(EXISTS "${GLM_TIERSTREAM_SRC}/CMakeLists.txt" AND EXISTS "${GLM_TIERSTREAM_SRC}/include/tierstream/ts.h")
  include(ExternalProject)
  set(GLM_TS_HAVE ON)
  set(GLM_TS_COMMIT "")
  if(EXISTS "${GLM_TIERSTREAM_SRC}/STAMP")
    file(STRINGS "${GLM_TIERSTREAM_SRC}/STAMP" GLM_TS_COMMIT LIMIT_COUNT 1)
  endif()
  string(LENGTH "${GLM_TIERSTREAM_PIN}" GLM_TS_PIN_LEN)
  string(LENGTH "${GLM_TS_COMMIT}" GLM_TS_COMMIT_LEN)
  set(GLM_TS_PREFIX "")
  if(GLM_TS_COMMIT_LEN EQUAL 40)
    string(SUBSTRING "${GLM_TS_COMMIT}" 0 ${GLM_TS_PIN_LEN} GLM_TS_PREFIX)
  endif()
  if(NOT GLM_TS_PREFIX STREQUAL GLM_TIERSTREAM_PIN)
    message(FATAL_ERROR "GLM_TIERSTREAM_SRC ${GLM_TIERSTREAM_SRC}: STAMP '${GLM_TS_COMMIT}' is not the pinned ${GLM_TIERSTREAM_PIN} (export with git archive and write the full commit to STAMP)")
  endif()
  foreach(flavor cuda cpu)
    if(flavor STREQUAL "cuda")
      set(ts_cuda ON)
    else()
      set(ts_cuda OFF)
    endif()
    # The sub-build runs as many jobs as the producer build (build.ps1 -Parallel), default one.
    set(GLM_TS_PARALLEL 1 CACHE STRING "parallel jobs of the tierstream sub-build")
    set(ts_dir "${CMAKE_BINARY_DIR}/tierstream-${flavor}")
    ExternalProject_Add(tierstream_${flavor}
      SOURCE_DIR "${GLM_TIERSTREAM_SRC}" BINARY_DIR "${ts_dir}" DOWNLOAD_COMMAND "" UPDATE_COMMAND "" INSTALL_COMMAND ""
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DTS_CUDA=${ts_cuda} -DTS_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120
                 -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
                 -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_CUDA_COMPILER_LAUNCHER=
      BUILD_COMMAND ${CMAKE_COMMAND} --build "${ts_dir}" --target tierstream --parallel ${GLM_TS_PARALLEL}
      BUILD_BYPRODUCTS "${ts_dir}/tierstream.lib")
    add_library(glm_tierstream_${flavor} STATIC IMPORTED)
    set_target_properties(glm_tierstream_${flavor} PROPERTIES IMPORTED_LOCATION "${ts_dir}/tierstream.lib")
    add_dependencies(glm_tierstream_${flavor} tierstream_${flavor})
  endforeach()
  add_library(producer_tierstream STATIC tierstream_binding.cpp budget_adapter.cpp)
  target_include_directories(producer_tierstream PRIVATE "${GLM_TIERSTREAM_SRC}/include")
  add_dependencies(producer_tierstream tierstream_cuda)
  target_link_libraries(glm_native_service PRIVATE producer_tierstream glm_tierstream_cuda avrt winmm bcrypt)
  add_library(producer_tierstream_cpu STATIC tierstream_binding.cpp budget_adapter.cpp)
  target_include_directories(producer_tierstream_cpu PRIVATE "${GLM_TIERSTREAM_SRC}/include")
  target_compile_definitions(producer_tierstream PRIVATE GLM_TIERSTREAM_COMMIT="${GLM_TS_COMMIT}")
  target_compile_definitions(producer_tierstream_cpu PRIVATE GLM_TIERSTREAM_COMMIT="${GLM_TS_COMMIT}")
  add_dependencies(producer_tierstream_cpu tierstream_cpu)
  target_link_libraries(glm_native_decode_host PRIVATE producer_tierstream_cpu glm_tierstream_cpu avrt winmm bcrypt)
  target_compile_definitions(glm_native_decode_host PRIVATE GLM_TIERSTREAM_LIBRARY=1)
else()
  add_library(producer_tierstream STATIC tierstream_stub.cpp)
  target_link_libraries(glm_native_service PRIVATE producer_tierstream)
endif()
add_executable(glm_unit_file unit_file_tool.cpp)
target_link_libraries(glm_unit_file PRIVATE glm_host_hash glm_json)
if(GLM_TS_HAVE)
  # Descriptor test: the real desc-v1 parser from the CPU library (internal header).
  target_sources(glm_native_decode_host PRIVATE "${REPO}/tests/glm_native/tierstream_desc_probe.cpp")
  set_source_files_properties("${REPO}/tests/glm_native/tierstream_desc_probe.cpp" PROPERTIES INCLUDE_DIRECTORIES "${GLM_TIERSTREAM_SRC}/include;${GLM_TIERSTREAM_SRC}/src")
  target_compile_definitions(glm_native_decode_host PRIVATE GLM_TIERSTREAM_PARSER=1)
else()
  message(WARNING "GLM_TIERSTREAM_SRC has no tierstream sources; tierstream backend refused and its CPU checks skipped")
endif()
