add_executable(PCA PCAtest.cpp PCA.h)
target_link_libraries(PCA PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_compile_options(PCA PRIVATE -O3 -march=native)
