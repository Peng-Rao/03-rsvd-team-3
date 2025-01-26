add_executable(speedup_rsvd speedup_rsvd.cpp)
target_link_libraries(speedup_rsvd PRIVATE Eigen3::Eigen)
target_link_libraries(speedup_rsvd PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(speedup_rsvd PRIVATE -O3 -march=native)