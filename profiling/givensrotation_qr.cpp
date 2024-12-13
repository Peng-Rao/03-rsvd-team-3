#include "GivensRotationQR.h"

#include <Eigen/Core>
#include <iostream>


int main() {
    // Eigen::Matrix<double, 3, 3> A;
    // A << 12, -51,   4,
    //       6, 167, -68,
    //      -4,  24, -41;

    // Define SparseMatrix RowMajor
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(3, 3);
    A.insert(0, 0) = 12;
    A.insert(0, 1) = -51;
    A.insert(0, 2) = 4;
    A.insert(1, 0) = 6;
    A.insert(1, 1) = 167;
    A.insert(1, 2) = -68;
    A.insert(2, 0) = -4;
    A.insert(2, 1) = 24;
    A.insert(2, 2) = -41;
    std::cout << "Q:\n" << A << "\n\n";
    A.makeCompressed();
    Eigen::GivensRotationQR<Eigen::SparseMatrix<double, Eigen::RowMajor>> qr;
    // Eigen::GivensRotationQR<Eigen::MatrixXd> qr;
    qr.compute(A);

    Eigen::MatrixXd Q = qr.matrixQ();
    Eigen::MatrixXd R = qr.matrixR();

    std::cout << "Q:\n" << Q << "\n\n";
    std::cout << "R:\n" << R << "\n\n";
    std::cout << "Reconstructed A (Q*R):\n" << Q * R << "\n\n";

    return 0;
}