//#define EIGEN_USE_BLAS

#include "RandomizedSVD.h"
#include <Eigen/Dense>

#include <iostream>


int main() {
    // Generate a random dense matrix
    int n {100};
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    int k = 100;

    // Perform Randomized SVD
    Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;
    rsvd.compute(A, k);

    // calculate the approximation error
    const Eigen::MatrixXd A_approx = rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().transpose();
    const double frobenius_norm = (A - A_approx).norm();
    std::cout << "Frobenius norm of approximation error: " << frobenius_norm << std::endl;
    return 0;
}