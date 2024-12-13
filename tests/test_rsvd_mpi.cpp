
#include "RandomizedSVD.h"


#include <catch2/catch_test_macros.hpp>

TEST_CASE("RandomizedSVD on Hilbert matrix", "[rsvd_hilbert]") {
    using Scalar = double;
    using MatrixXd = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // set the dimensions of the Hilbert matrix
    constexpr int n{100};
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

TEST_CASE("RandomizedSVD MPI on Hilbert matrix", "[rsvd_mpi_hilbert]") {
    using Scalar = double;
    using MatrixXd = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // set the dimensions of the Hilbert matrix
    constexpr int n{100};
    constexpr int rank{10};
    constexpr int powerIterations{2};

    // create the Hilbert matrix
    MatrixXd H(n, n);
    for (int i{0}; i < n; ++i) {
        for (int j{0}; j < n; ++j) {
            H(i, j) = 1.0 / (static_cast<double>(i + j + 1));
        }
    }

    // Get MPI info
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // compute the RSVD using MPI
    Eigen::RandomizedSVD<MatrixXd> rsvd;
    rsvd.compute_mpi(H, rank, powerIterations);

    // Only rank 0 performs the verification
    if (world_rank == 0) {
        auto U_rsvd = rsvd.matrixU();
        auto S_rsvd = rsvd.singularValues();
        auto V_rsvd = rsvd.matrixV();

        // Verify reconstruction
        Eigen::MatrixXd S_diag_rsvd = S_rsvd.asDiagonal();
        Eigen::MatrixXd H_approx = U_rsvd.leftCols(rank) * S_diag_rsvd.topLeftCorner(rank, rank) * V_rsvd.leftCols(rank).transpose();
        double reconstruction_error = (H - H_approx).norm() / H.norm();
        REQUIRE(reconstruction_error < 1e-5);
    }

    // Finalize MPI
    MPI_Finalize();
}
#include "RandomizedSVD.h"
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <mpi.h>

TEST_CASE("RandomizedSVD MPI on Hilbert matrix", "[rsvd_mpi_hilbert]") {
    using Scalar = double;
    using MatrixXd = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // set the dimensions of the Hilbert matrix
    constexpr int n{100};
    constexpr int rank{10};
    constexpr int powerIterations{2};

    // create the Hilbert matrix
    MatrixXd H(n, n);
    for (int i{0}; i < n; ++i) {
        for (int j{0}; j < n; ++j) {
            H(i, j) = 1.0 / (static_cast<double>(i + j + 1));
        }
    }

    // Initialize MPI
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(NULL, NULL);
    }

    // Get the number of processes and the rank of this process
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Distribute the matrix rows among processes
    int local_rows = n / world_size;
    int start_row = world_rank * local_rows;
    int end_row = (world_rank == world_size - 1) ? n : start_row + local_rows;
    MatrixXd local_H = H.block(start_row, 0, end_row - start_row, n);

    // compute the RSVD using MPI
    Eigen::RandomizedSVD<MatrixXd> rsvd;
    rsvd.compute_mpi(local_H, rank, powerIterations);
    auto U_rsvd = rsvd.matrixU();
    auto S_rsvd = rsvd.singularValues();
    auto V_rsvd = rsvd.matrixV();

    // Gather results from all processes
    MatrixXd global_U(n, rank);
    MatrixXd global_V(n, rank);
    Eigen::VectorXd global_S(rank);

    MPI_Gather(U_rsvd.data(), local_rows * rank, MPI_DOUBLE, global_U.data(), local_rows * rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(V_rsvd.data(), local_rows * rank, MPI_DOUBLE, global_V.data(), local_rows * rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(S_rsvd.data(), rank, MPI_DOUBLE, global_S.data(), rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // JacobiSVD as a reference
        Eigen::JacobiSVD svd_ref(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto S_ref = svd_ref.singularValues();

        Eigen::MatrixXd S_diag_rsvd = global_S.asDiagonal();
        Eigen::MatrixXd H_approx =
                global_U.leftCols(rank) * S_diag_rsvd.topLeftCorner(rank, rank) * global_V.leftCols(rank).transpose();

        // Check the reconstruction error of the Hilbert matrix
        double reconstruction_error = (H - H_approx).norm() / H.norm();
        INFO("Reconstruction error (relative) for Hilbert matrix: " << reconstruction_error);
        REQUIRE(reconstruction_error < 1e-5);

        // The singular values of the Hilbert matrix decay rapidly, check the proportion of the first rank singular values.
        for (int i = 0; i < rank; ++i) {
            double ratio = global_S(i) / S_ref(i);
            INFO("Singular value ratio at " << i << " is " << ratio);
            REQUIRE(ratio > 0.1);
            REQUIRE(ratio < 10.0);
        }

        // Orthogonality check (U, V should be approximately orthogonal)
        Eigen::MatrixXd UU = global_U.transpose() * global_U;
        double orth_err_U = (UU - MatrixXd::Identity(UU.rows(), UU.cols())).norm();
        INFO("Orthogonality error of U for Hilbert: " << orth_err_U);
        REQUIRE(orth_err_U < 1e-6);

        Eigen::MatrixXd VV = global_V.transpose() * global_V;
        double orth_err_V = (VV - MatrixXd::Identity(VV.rows(), VV.cols())).norm();
        INFO("Orthogonality error of V for Hilbert: " << orth_err_V);
        REQUIRE(orth_err_V < 1e-6);
    }

    // Finalize MPI
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        MPI_Finalize();
    }
}
