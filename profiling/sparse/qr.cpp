#include "GivensRotationQR.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
    // Define matrix size and sparsity
    constexpr int matrixSize = 5000;
    constexpr double sparsity = 0.1; // Fraction of non-zero elements

    // Random number generator for matrix elements
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution valueDist(-1.0, 1.0); // Values for matrix elements
    std::uniform_real_distribution sparsityDist(0.0, 1.0); // To determine sparsity

    // Generate a random sparse matrix
    Eigen::SparseMatrix<double> sparseMatrix(matrixSize, matrixSize);
    std::vector<Eigen::Triplet<double>> triplets;

    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            if (sparsityDist(gen) < sparsity) {
                triplets.emplace_back(i, j, valueDist(gen));
            }
        }
    }

    sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());

    // Perform Sparse QR Decomposition
    Eigen::GivensRotationQR<Eigen::SparseMatrix<double>> givens_rotation_qr;
    // Eigen::setNbThreads(4);
    auto start_time = std::chrono::high_resolution_clock::now();
    givens_rotation_qr.compute(sparseMatrix);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "QR decomposition time: " << duration.count() << " ms" << std::endl;

    Eigen::MatrixXd Q = givens_rotation_qr.matrixQ();
    Eigen::MatrixXd R = givens_rotation_qr.matrixR();
    Eigen::MatrixXd QR = Q * R;

    double frobenius_norm = (sparseMatrix - QR).norm();
    std::cout << "Frobenius err norm: " << frobenius_norm << std::endl;
    return 0;
}









