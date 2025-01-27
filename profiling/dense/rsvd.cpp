// #define EIGEN_USE_BLAS

#include "RandomizedSVD.h"

#include <Eigen/Dense>

#include <iostream>


int main() {
    // Generate a low rank random dense matrix
    const int rows = 1000;
    const int cols = 1000;
    const int low_rank = 50;
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(rows, low_rank);
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(cols, low_rank);

    // Construct the low-rank matrix A
    Eigen::MatrixXd A = U * V.transpose();

    int k = 50;
    // Perform Randomized SVD
    Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;
    rsvd.compute(A, k);

    // calculate the approximation error
    const Eigen::MatrixXd A_approx = rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().transpose();
    const double frobenius_norm = (A - A_approx).norm() / A.norm();
    std::cout << "Frobenius norm of approximation error: " << frobenius_norm << std::endl;
    return 0;
}
