# MiniMax H3 shared host library (types and fixture IO). Packages link strix_h3_common.
add_library(strix_h3_common INTERFACE)
target_include_directories(strix_h3_common INTERFACE src include)
target_compile_features(strix_h3_common INTERFACE cxx_std_20)
if(NOT MSVC)
    target_compile_options(strix_h3_common INTERFACE -fno-fast-math -ffp-contract=off)
endif()
