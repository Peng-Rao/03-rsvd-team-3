#include "RandomizedSVD_OMP.h"
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
    std::cout << "Starting RSVD computation..." << std::endl;
    setEigenThreads(8);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(2000, 2000);
    rsvd.compute(A, 100, 2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "RSVD mpi took " << elapsed.count() << " seconds." << std::endl;
    return 0;
}