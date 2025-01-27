add_executable(mpi_random_matrix mpi_random_matrix.cpp)
target_link_libraries(mpi_random_matrix PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_link_libraries(mpi_random_matrix PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(mpi_random_matrix PRIVATE OpenBLAS::OpenBLAS)
target_compile_options(mpi_random_matrix PRIVATE -O3 -march=native)

add_executable(omp_random_matrix omp_random_matrix.cpp)
target_link_libraries(omp_random_matrix PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_link_libraries(omp_random_matrix PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(omp_random_matrix PRIVATE OpenBLAS::OpenBLAS)
target_compile_options(omp_random_matrix PRIVATE -O3 -march=native)


add_executable(mpi_omp_random_matrix mpi_omp_random_matrix.cpp)
target_link_libraries(mpi_omp_random_matrix PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_link_libraries(mpi_omp_random_matrix PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(mpi_omp_random_matrix PRIVATE OpenBLAS::OpenBLAS)
target_compile_options(mpi_omp_random_matrix PRIVATE -O3 -march=native)

add_executable(mpi_rsvd mpi_rsvd.cpp)
target_link_libraries(mpi_rsvd PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_link_libraries(mpi_rsvd PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(mpi_rsvd PRIVATE OpenBLAS::OpenBLAS)
target_compile_options(mpi_rsvd PRIVATE -O3 -march=native)

add_subdirectory(dense)
add_subdirectory(sparse)