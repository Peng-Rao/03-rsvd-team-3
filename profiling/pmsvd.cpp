#include "PowerMethodSVD.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>

int main() {
    Eigen::MatrixXd A(4, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10,11,12;

    int rank = 2;

    Eigen::PowerMethodSVD<Eigen::MatrixXd> pmsvd;

    pmsvd.compute(A, rank, 1000, 1e-6);

    Eigen::VectorXd sing_vals = pmsvd.singularValues();
    Eigen::MatrixXd U = pmsvd.matrixU();
    Eigen::MatrixXd V = pmsvd.matrixV();

    std::cout << "Singular values:\n" << sing_vals << "\n\n";
    std::cout << "U:\n" << U << "\n\n";
    std::cout << "V:\n" << V << "\n\n";

    // sparse matrix
    Eigen::SparseMatrix<double> As = A.sparseView();
    Eigen::PowerMethodSVD<Eigen::SparseMatrix<double>> pmsvds;

    pmsvds.compute(As, rank, 1000, 1e-6);

    Eigen::VectorXd sing_valss = pmsvds.singularValues();
    Eigen::MatrixXd Us = pmsvds.matrixU();
    Eigen::MatrixXd Vs = pmsvds.matrixV();

    std::cout << "Singular values:\n" << sing_valss << "\n\n";
    std::cout << "U:\n" << Us << "\n\n";
    std::cout << "V:\n" << Vs << "\n\n";

    return 0;
}