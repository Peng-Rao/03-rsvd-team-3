#define EIGEN_USE_BLAS

#include "RandomizedSVD.h"
#include <Eigen/Dense>


int main() {
    constexpr int n{2000};
    constexpr int rank{10};
    constexpr int powerIterations{2};

    const Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;
    rsvd.compute(A, rank, powerIterations);
    return 0;
}