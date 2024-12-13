#include <mpi.h>
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <chrono>
#include "RandomizedSVD_MPI.h"

// Test different matrix sizes for better comparison
std::vector<int> sizes = {10000};
constexpr int rank{10};
constexpr int powerIterations{2};

TEST_CASE("Custom RSVD MPI Implementation Benchmark", "[custom_rsvd_mpi_bench]") {
    // Initialize MPI
    MPI_Init(NULL, NULL);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    for (int size : sizes) {
        // Create test matrix for each size (same on all processes)
        Eigen::MatrixXd testMatrix = Eigen::MatrixXd::Random(size, size);
        
        // Synchronize before benchmarking
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Benchmark MPI version
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;
            Eigen::setNbThreads(8);
            rsvd.compute(testMatrix, rank, powerIterations);
            
            // Only synchronize after MPI computation
            MPI_Barrier(MPI_COMM_WORLD);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            if (world_rank == 0) {
                std::cout << "Matrix size: " << size << "x" << size << std::endl;
                std::cout << "MPI RSVD with " << world_size << " processes took: " 
                         << elapsed.count() << " seconds" << std::endl;
                std::cout << "-------------------" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Process " << world_rank << " failed: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Finalize MPI
    MPI_Finalize();
}
