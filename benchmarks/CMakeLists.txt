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

# config for my rsvd benchmarks
add_executable(BenchMyRSVD bench_myRSVD.cpp)
target_link_libraries(BenchMyRSVD PRIVATE Catch2::Catch2WithMain)
target_link_libraries(BenchMyRSVD PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(BenchMyRSVD PRIVATE -O3)