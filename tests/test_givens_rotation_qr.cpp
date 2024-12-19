#include "GivensRotationQR.h"

#include <Eigen/Sparse>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>

using Eigen::SparseMatrix;
using Eigen::MatrixXd;

TEST_CASE("Givens Rotation on Sparse Random Matrices with Error Reporting", "[rsvd_sparse_random_error]") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(-100.0, 100.0);

    for (int size = 100; size <= 1000; size *= 2) {
        // Define sparsity (percentage of non-zero elements)
        constexpr double sparsity = 0.01; // 1% non-zero entries

        // Generate sparse random matrix
        SparseMatrix<double> A(size, size);
        std::vector<Eigen::Triplet<double>> triplets;
        const int nonZeroEntries = static_cast<int>(size * size * sparsity);

        for (int i = 0; i < nonZeroEntries; ++i) {
            int row = gen() % size;
            int col = gen() % size;
            double value = dis(gen);
            triplets.emplace_back(row, col, value);
        }
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Convert SparseMatrix to Dense for GivensRotationQR
        // auto A_dense = MatrixXd(A);

        // Perform GivensRotationQR decomposition
        Eigen::GivensRotationQR<SparseMatrix<double>> qr;
        qr.compute(A);

        // Get Q and R matrices
        MatrixXd Q = qr.matrixQ();
        MatrixXd R = qr.matrixR();

        // Calculate reconstruction error: ||A - Q * R||_F / ||A||_F
        MatrixXd reconstructedA = Q * R;
        const double frobeniusNormOriginal = A.norm();
        const double frobeniusNormError = (A - reconstructedA).norm();
        const double relativeError = frobeniusNormError / frobeniusNormOriginal;

        // Log size and error
        std::cout << "Matrix size: " << size << "x" << size
                  << ", Sparsity: " << sparsity
                  << ", Relative Error: " << relativeError << std::endl;

        // Check if relative error is less than 1e-1
        REQUIRE(relativeError < 1e-1);
    }
}