cmake_minimum_required(VERSION 3.20)
project(03_rsvd_team_3)

set(CMAKE_CXX_STANDARD 20)
enable_testing()

find_package(Eigen3 CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(OpenBLAS CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(SYSTEM ${MPI_INCLUDE_DIRS})

include(CTest)
include(Catch)

add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(profiling)
add_subdirectory(applications)
