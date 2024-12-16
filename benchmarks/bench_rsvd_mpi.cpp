#include "RandomizedSVD_MPI.h"
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <mpi.h>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

// constexpr int n{8000};
// constexpr int rank{100};
// constexpr int powerIterations{2};

// Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

void setEigenThreads(int numThreads) {
    Eigen::setNbThreads(numThreads);
}

TEST_CASE("RSVD mpi", "[rsvd_mpi]") {
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Init(nullptr, nullptr);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        std::cout << "Starting RSVD computation..." << std::endl;
    }

    setEigenThreads(1);
    
    rsvd.generateRandomMatrix(1000, 1000);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (world_rank == 0) {
        std::cout << "RSVD mpi took " << elapsed.count() << " seconds." << std::endl;
    }

    MPI_Finalize();
}