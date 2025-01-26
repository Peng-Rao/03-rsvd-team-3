#include "RandomizedSVD.h"
#include "PowerMethodSVD.h"

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

int main()
{
    // -------------------------------------------------------------------------
    // 1) Parameter Setup
    // -------------------------------------------------------------------------
    // Matrix dimensions
    const int rows = 1000;
    const int cols = 1000;

    // The true rank of the generated low-rank matrix
    // We'll construct A = U * V^T with rank up to "low_rank"
    const int low_rank = 200;

    // Different values of k for partial SVD
    // Even if the actual rank is 50, we test values > 50 to see the behavior
    std::vector<int> ranks = {5, 20, 50, 100, 200};

    // -------------------------------------------------------------------------
    // 2) Generate a low-rank matrix A
    // -------------------------------------------------------------------------
    // U: 1000 x 50
    // V: 1000 x 50
    // A = U * V^T => 1000 x 1000, with rank <= 50
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(rows, low_rank);
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(cols, low_rank);

    // Construct the low-rank matrix A
    Eigen::MatrixXd A = U * V.transpose();

    // Compute the Frobenius norm of A for relative error calculation
    double A_norm = A.norm();

    // Open a CSV file to store the results
    std::ofstream csv_file("svd_results.csv");
    if (!csv_file.is_open())
    {
        std::cerr << "Error opening the CSV file!" << std::endl;
        return -1;
    }

    // Write CSV header
    csv_file << "Method,k,Time(s),Speedup,RelativeFroError\n";

    // -------------------------------------------------------------------------
    // 3) Test each k
    // -------------------------------------------------------------------------
    for (int k : ranks)
    {
        std::cout << ">>> Processing k = " << k << " ..." << std::endl;

        // ===========================================================
        // 3.1) BDCSVD (Eigen's built-in)
        // ===========================================================
        auto start = std::chrono::high_resolution_clock::now();

        // Use BDCSVD to compute the SVD
        Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_bdc = end - start;
        double bdc_time = elapsed_bdc.count();

        // Construct the rank-k approximation from BDCSVD
        // If k exceeds the true rank, there's no further improvement in practice
        const auto &U_bdc = svd.matrixU().leftCols(k);
        const auto &V_bdc = svd.matrixV().leftCols(k);
        Eigen::VectorXd S_bdc = svd.singularValues().head(k);

        Eigen::MatrixXd A_approx_bdc = U_bdc * S_bdc.asDiagonal() * V_bdc.transpose();
        double bdc_error = (A - A_approx_bdc).norm() / A_norm;

        // BDCSVD is our reference, so speedup = 1.0
        csv_file << "BDCSVD," << k << ","
                 << bdc_time << ","
                 << 1.0 << ","
                 << bdc_error << "\n";

        // Power Method SVD
        start = std::chrono::high_resolution_clock::now();

        Eigen::PowerMethodSVD<Eigen::MatrixXd> pmsvd;
        pmsvd.compute(A, k);

        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_pm = end - start;
        double pm_time = elapsed_pm.count();

        Eigen::MatrixXd A_approx_pm = pmsvd.matrixU() *
                                     pmsvd.singularValues().asDiagonal() *
                                     pmsvd.matrixV().transpose();
        double pm_error = (A - A_approx_pm).norm() / A_norm;

        double pm_speedup = bdc_time / pm_time;
        csv_file << "PowerMethodSVD," << k << ","
                 << pm_time << ","
                 << pm_speedup << ","
                 << pm_error << "\n";

        // Randomized SVD (1 power iteration)
        {
            start = std::chrono::high_resolution_clock::now();

            Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd_iter1;
            // compute(A, rank, power_iterations)
            rsvd_iter1.compute(A, k, 1);

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_rsvd1 = end - start;
            double rsvd1_time = elapsed_rsvd1.count();

            // Build the approximate matrix
            Eigen::MatrixXd A_approx_rsvd1 = rsvd_iter1.matrixU() *
                                             rsvd_iter1.singularValues().asDiagonal() *
                                             rsvd_iter1.matrixV().transpose();
            double rsvd1_error = (A - A_approx_rsvd1).norm() / A_norm;

            double rsvd1_speedup = bdc_time / rsvd1_time;
            csv_file << "RandomizedSVD_Iter1," << k << ","
                     << rsvd1_time << ","
                     << rsvd1_speedup << ","
                     << rsvd1_error << "\n";
        }

        // ===========================================================
        // 3.4) Randomized SVD (2 power iterations)
        // ===========================================================
        {
            start = std::chrono::high_resolution_clock::now();

            Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd_iter2;
            rsvd_iter2.compute(A, k, 2);

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_rsvd2 = end - start;
            double rsvd2_time = elapsed_rsvd2.count();

            Eigen::MatrixXd A_approx_rsvd2 = rsvd_iter2.matrixU() *
                                             rsvd_iter2.singularValues().asDiagonal() *
                                             rsvd_iter2.matrixV().transpose();
            double rsvd2_error = (A - A_approx_rsvd2).norm() / A_norm;

            double rsvd2_speedup = bdc_time / rsvd2_time;
            csv_file << "RandomizedSVD_Iter2," << k << ","
                     << rsvd2_time << ","
                     << rsvd2_speedup << ","
                     << rsvd2_error << "\n";
        }

        std::cout << "Done for k = " << k << "\n" << std::endl;
    }

    // Close the CSV file
    csv_file.close();
    std::cout << "Results saved to svd_results.csv" << std::endl;

    return 0;
}