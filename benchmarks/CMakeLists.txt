add_executable(BenchHouseholder bench_householder.cpp)
target_link_libraries(BenchHouseholder PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchHouseholder PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(BenchHouseholder PRIVATE -O3 -march=native)

add_subdirectory(dense)
add_subdirectory(sparse)