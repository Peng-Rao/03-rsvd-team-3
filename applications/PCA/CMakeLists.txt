add_executable(PCA PCA.cpp)
target_link_libraries(PCA PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_compile_options(PCA PRIVATE -O3 -march=native)
