#include "RandomizedSVD.h"

#include <iostream>
#include <mpi.h>
#include <chrono>



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int num_procs, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();


    Eigen::MatrixXd A(3, 3);
    A << 1, 3, 2, 5, 3, 1, 3, 4, 5;

    Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

    constexpr int rank{2};
    constexpr int powerIter{3};

    rsvd.compute(A, rank, powerIter);

    const auto &singularValues = rsvd.singularValues();
    const auto &U = rsvd.matrixU();
    const auto &V = rsvd.matrixV();

    std::cout << "Singular values: " << singularValues.transpose() << std::endl;
    std::cout << "U matrix: " << U << std::endl;
    std::cout << "V matrix: " << V << std::endl;



    // Record the end time
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // Print the execution time
    // if (mpi_rank == 0) {
    //     std::cout << "Execution time: " << duration << " milliseconds." << std::endl;
    // }
    std::cout << "Execution time: " << duration << " milliseconds." << std::endl;

    MPI_Finalize();
    return 0;
}

