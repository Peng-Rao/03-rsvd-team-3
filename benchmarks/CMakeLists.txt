
# config for MPI RSVD benchmarks
add_executable(BenchRSVDMPI bench_rsvd_mpi.cpp)
target_link_libraries(BenchRSVDMPI PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchRSVDMPI PRIVATE Eigen3::Eigen MPI::MPI_CXX)
target_compile_options(BenchRSVDMPI PRIVATE -O3)

add_executable(BenchHouseholder bench_householder.cpp)
target_link_libraries(BenchHouseholder PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchHouseholder PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(BenchHouseholder PRIVATE -O3 -march=native)

add_executable(BenchGivensRotation bench_givensrotation.cpp)
target_link_libraries(BenchGivensRotation PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchGivensRotation PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(BenchGivensRotation PRIVATE -O3 -march=native)

add_executable(BenchRandomizedSVD bench_randomizedSVD.cpp)
target_link_libraries(BenchRandomizedSVD PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchRandomizedSVD PRIVATE Eigen3::Eigen)
target_link_libraries(BenchRandomizedSVD PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(BenchRandomizedSVD PRIVATE -O3 -march=native)

add_executable(BenchPowerMethod bench_powermethod.cpp)
target_link_libraries(BenchPowerMethod PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchPowerMethod PRIVATE Eigen3::Eigen)
target_link_libraries(BenchPowerMethod PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(BenchPowerMethod PRIVATE -O3 -march=native)