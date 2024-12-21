#include "RandomizedSVD_MPI.h"
#include <Eigen/Dense>
#include <mpi.h>
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

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Init(nullptr, nullptr);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        std::cout << "Starting RSVD computation..." << std::endl;
    }

    setEigenThreads(1);
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(2000, 2000);
    rsvd.compute(A, 100, 2);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (world_rank == 0) {
        std::cout << "RSVD mpi took " << elapsed.count() << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}