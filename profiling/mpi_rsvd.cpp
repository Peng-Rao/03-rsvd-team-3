#include "RandomizedSVD_MPI.h"
#include <Eigen/Dense>
#include <mpi.h>
#include <chrono>
#include <iostream>

int n=10000;
int rank=100;
int powerIterations=2;

void setEigenThreads(int numThreads) {
    Eigen::setNbThreads(numThreads);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    setEigenThreads(1);

    Eigen::MatrixXd A;
    if (world_rank == 0) {
        std::cout << "Starting RSVD computation..." << std::endl;
        A = Eigen::MatrixXd::Random(n, n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        A.resize(n, n);
    }
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

    auto start = std::chrono::high_resolution_clock::now();
    rsvd.compute(A, rank, powerIterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Process " << world_rank << " RSVD computation took " << elapsed.count() << " seconds." << std::endl;

    MPI_Finalize();
    return 0;
}