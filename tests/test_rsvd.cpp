#include "RandomizedSVD.h"

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("RandomizedSVD on Hilbert matrix", "[rsvd_hilbert]") {
    using Scalar = double;
    using MatrixXd = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // set the dimensions of the Hilbert matrix
    constexpr int n{500};
    constexpr int rank{10};
    constexpr int powerIterations{2};

    // create the Hilbert matrix
    MatrixXd H(n, n);
    for (int i{0}; i < n; ++i) {
        for (int j{0}; j < n; ++j) {
            H(i, j) = 1.0 / (static_cast<double>(i + j + 1));
        }
    }

    // compute the RSVD
    Eigen::RandomizedSVD<MatrixXd> rsvd;
    rsvd.compute(H, rank, powerIterations);
    auto U_rsvd = rsvd.matrixU();
    auto S_rsvd = rsvd.singularValues();
    auto V_rsvd = rsvd.matrixV();

    // JacobiSVD as a reference
    Eigen::JacobiSVD svd_ref(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto S_ref = svd_ref.singularValues();

    Eigen::MatrixXd S_diag_rsvd = S_rsvd.asDiagonal();
    Eigen::MatrixXd H_approx =
            U_rsvd.leftCols(rank) * S_diag_rsvd.topLeftCorner(rank, rank) * V_rsvd.leftCols(rank).transpose();


    // Check the reconstruction error of the Hilbert matrix
    double reconstruction_error = (H - H_approx).norm() / H.norm();
    INFO("Reconstruction error (relative) for Hilbert matrix: " << reconstruction_error);

    REQUIRE(reconstruction_error < 1e-5);


    // The singular values of the Hilbert matrix decay rapidly, check the proportion of the first rank singular values.
    for (int i = 0; i < rank; ++i) {
        double ratio = S_rsvd(i) / S_ref(i);
        INFO("Singular value ratio at " << i << " is " << ratio);
        REQUIRE(ratio > 0.1);
        REQUIRE(ratio < 10.0);
    }

    // Orthogonality check (U, V should be approximately orthogonal)
    Eigen::MatrixXd UU = U_rsvd.transpose() * U_rsvd;
    double orth_err_U = (UU - MatrixXd::Identity(UU.rows(), UU.cols())).norm();
    INFO("Orthogonality error of U for Hilbert: " << orth_err_U);
    REQUIRE(orth_err_U < 1e-6);

    Eigen::MatrixXd VV = V_rsvd.transpose() * V_rsvd;
    double orth_err_V = (VV - MatrixXd::Identity(VV.rows(), VV.cols())).norm();
    INFO("Orthogonality error of V for Hilbert: " << orth_err_V);
    REQUIRE(orth_err_V < 1e-6);
}

