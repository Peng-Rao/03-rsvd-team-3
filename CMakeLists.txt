cmake_minimum_required(VERSION 3.31)
project(03_rsvd_team_3)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)

add_executable(main
    src/main.cpp)

target_link_libraries(main PRIVATE Eigen3::Eigen)
target_link_libraries(main PRIVATE Catch2::Catch2 Catch2::Catch2WithMain)