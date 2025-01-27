add_executable(dense_svd dense_svd.cpp)
target_link_libraries(dense_svd PRIVATE Eigen3::Eigen)
target_link_libraries(dense_svd PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(dense_svd PRIVATE -O3 -march=native)