# MiniMax H3 reference fixtures: reader test (host only; fixtures themselves live outside git).
add_executable(strix-h3-fixture-reader-tests tests/h3/fixtures/fixture_reader_tests.cpp)
target_link_libraries(strix-h3-fixture-reader-tests PRIVATE strix_h3_common)
add_test(NAME strix-h3-fixture-reader-tests COMMAND strix-h3-fixture-reader-tests)
