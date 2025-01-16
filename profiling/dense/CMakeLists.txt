add_executable(rsvd_dense_profiling rsvd.cpp)
target_link_libraries(rsvd_dense_profiling PRIVATE Eigen3::Eigen)
target_link_libraries(rsvd_dense_profiling PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(rsvd_dense_profiling PRIVATE -O3 -march=native)