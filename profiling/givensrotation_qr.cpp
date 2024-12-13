#include "GivensRotationQR.h"

#include <Eigen/Core>

#include <iostream>


int main() {
    Eigen::Matrix<double, 3, 3> A;
    A << 12, -51,   4,
          6, 167, -68,
         -4,  24, -41;

    Eigen::GivensRotationQR<decltype(A)> qr;
    qr.compute(A);

    Eigen::MatrixXd Q = qr.matrixQ();
    Eigen::MatrixXd R = qr.matrixR();

    std::cout << "Q:\n" << Q << "\n\n";
    std::cout << "R:\n" << R << "\n\n";

    std::cout << "Reconstructed A (Q*R):\n" << Q * R << "\n\n";

    return 0;
}