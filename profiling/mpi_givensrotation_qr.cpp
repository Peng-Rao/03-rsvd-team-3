#include "GivensRotationQR_MPI.h"
#include <vector>
#include <random>
#include <mpi.h>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Define matrix size and sparsity
    constexpr int matrixSize = 1000;
    constexpr double sparsity = 0.1;
    
    Eigen::SparseMatrix<double> sparseMatrix(matrixSize, matrixSize);

    

    if (rank == 0) {
        // Random number generation
        std::mt19937 gen(42);
        std::uniform_real_distribution valueDist(-1.0, 1.0);
        std::uniform_real_distribution sparsityDist(0.0, 1.0);

        // Generate matrix
        std::vector<Eigen::Triplet<double>> triplets;
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                if (sparsityDist(gen) < sparsity) {
                    triplets.emplace_back(i, j, valueDist(gen));
                }
            }
        }
        sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    // Broadcast matrix data to all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform parallel QR decomposition
    
    Eigen::GivensRotationQR<Eigen::SparseMatrix<double>> givens_rotation_qr;
    givens_rotation_qr.compute(sparseMatrix);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "QR decomposition time: " << duration.count() << " ms" << std::endl;

        Eigen::MatrixXd Q = givens_rotation_qr.matrixQ();
        Eigen::MatrixXd R = givens_rotation_qr.matrixR();
        Eigen::MatrixXd QR = Q * R;
    
        double frobenius_norm = (sparseMatrix - QR).norm();
        std::cout << "Frobenius err norm: " << frobenius_norm << std::endl;
    }

    MPI_Finalize();
    return 0;
}