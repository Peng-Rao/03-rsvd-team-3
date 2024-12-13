# config for matmul benchmarks
add_executable(BenchMatmul bench_matmul.cpp)
target_link_libraries(BenchMatmul PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchMatmul PUBLIC OpenMP::OpenMP_CXX)
#target_link_libraries(BenchMatmul PRIVATE OpenBLAS::OpenBLAS)
target_compile_options(BenchMatmul PRIVATE -O3 -march=native)

# config for rsvd using OpenMP benchmarks
add_executable(BenchRSVD bench_rsvd.cpp)
target_link_libraries(BenchRSVD PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchRSVD PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(BenchRSVD PRIVATE OpenBLAS::OpenBLAS)
target_compile_options(BenchRSVD PRIVATE -O3 -march=native)

# config for MPI RSVD benchmarks
add_executable(BenchRSVDMPI bench_myRSVD_mpi.cpp)
target_link_libraries(BenchRSVDMPI PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchRSVDMPI PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_compile_options(BenchRSVDMPI PRIVATE -O3)


