#define EIGEN_USE_BLAS

#include "RandomizedSVD.h"
#include <Eigen/Dense>

#include <iostream>


int main() {
    Eigen::MatrixXd A(4, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10,11,12;

    int rank = 2;

    Eigen::RandomizedSVD<Eigen::MatrixXd> pmsvd;

    pmsvd.compute(A, rank);

    Eigen::VectorXd sing_vals = pmsvd.singularValues();
    Eigen::MatrixXd U = pmsvd.matrixU();
    Eigen::MatrixXd V = pmsvd.matrixV();

    std::cout << "Singular values:\n" << sing_vals << "\n\n";
    std::cout << "U:\n" << U << "\n\n";
    std::cout << "V:\n" << V << "\n\n";

    // sparse matrix
    Eigen::SparseMatrix<double> As = A.sparseView();
    Eigen::RandomizedSVD<Eigen::SparseMatrix<double>> pmsvds;

    pmsvds.compute(As, rank);

    Eigen::VectorXd sing_valss = pmsvds.singularValues();
    Eigen::MatrixXd Us = pmsvds.matrixU();
    Eigen::MatrixXd Vs = pmsvds.matrixV();

    std::cout << "Singular values:\n" << sing_valss << "\n\n";
    std::cout << "U:\n" << Us << "\n\n";
    std::cout << "V:\n" << Vs << "\n\n";

    return 0;
}