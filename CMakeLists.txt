cmake_minimum_required(VERSION 3.20)
project(03_rsvd_team_3)

set(CMAKE_CXX_STANDARD 20)


find_package(Eigen3 CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(SYSTEM ${MPI_INCLUDE_DIRS})

# config for unit tests
add_executable(UnitTests tests/test_rsvd.cpp)
target_link_libraries(UnitTests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(UnitTests PRIVATE MPI::MPI_CXX)

# config for matmul benchmarks
add_executable(BenchMatmul benchmarks/bench_matmul.cpp)
target_link_libraries(BenchMatmul PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchMatmul PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(BenchMatmul PRIVATE -O3)

# config for rsvd using OpenMP benchmarks
add_executable(BenchRSVD benchmarks/bench_rsvd.cpp)
target_link_libraries(BenchRSVD PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchRSVD PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(BenchRSVD PRIVATE -O3)

# config for my rsvd benchmarks
add_executable(BenchMyRSVD benchmarks/bench_myRSVD.cpp)
target_link_libraries(BenchMyRSVD PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchMyRSVD PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(BenchMyRSVD PRIVATE -O3)

include(CTest)
include(Catch)
catch_discover_tests(UnitTests)