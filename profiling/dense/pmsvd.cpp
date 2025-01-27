// #define EIGEN_USE_BLAS

#include "PowerMethodSVD.h"

#include <Eigen/Dense>

#include <iostream>


int main() {
    // Generate a low rank random dense matrix
    const int rows = 100;
    const int cols = 100;
    const int low_rank = 50;
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(rows, low_rank);
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(cols, low_rank);

    // Construct the low-rank matrix A
    Eigen::MatrixXd A = U * V.transpose();

    int k = 20;
    // Perform Randomized SVD
    Eigen::PowerMethodSVD<Eigen::MatrixXd> pmsvd;
    pmsvd.compute(A, k, 1e-9, 0.05, 0.001);

    // calculate the approximation error
    const Eigen::MatrixXd A_approx = pmsvd.matrixU() * pmsvd.singularValues().asDiagonal() * pmsvd.matrixV().transpose();
    const double frobenius_norm = (A - A_approx).norm() / A.norm();
    std::cout << "Frobenius norm of approximation error: " << frobenius_norm << std::endl;
    return 0;
}
