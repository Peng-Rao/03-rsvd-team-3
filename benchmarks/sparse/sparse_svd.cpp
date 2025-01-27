#include "RandomizedSVD.h"
#include "PowerMethodSVD.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SVD>  // for BDCSVD
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

int main()
{
    const int rows = 1000;
    const int cols = 1000;

    const int low_rank = 200;

    std::vector<int> ranks = {5, 20, 50, 100, 200};
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(rows, low_rank);
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(cols, low_rank);
    Eigen::MatrixXd A_dense = U * V.transpose();


    Eigen::SparseMatrix<double> A = A_dense.sparseView(1e-6);

    double frob_sum = 0.0;
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
        {
            double val = it.value();
            frob_sum += val * val;
        }
    }
    double A_norm = std::sqrt(frob_sum);

    std::ofstream csv_file("svd_results_sparse.csv");
    if (!csv_file.is_open())
    {
        std::cerr << "Error opening the CSV file!" << std::endl;
        return -1;
    }
    csv_file << "Method,k,Time(s),Speedup,RelativeFroError\n";

    // -------------------------------------------------------------------------
    // 3) Test each k
    // -------------------------------------------------------------------------
    for (int k : ranks)
    {
        std::cout << ">>> Processing k = " << k << " ..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        Eigen::MatrixXd A_bdc = Eigen::MatrixXd(A);
        Eigen::BDCSVD<Eigen::MatrixXd> svd(A_bdc, Eigen::ComputeThinU | Eigen::ComputeThinV);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_bdc = end - start;
        double bdc_time = elapsed_bdc.count();

        const auto &U_bdc = svd.matrixU().leftCols(k);
        const auto &V_bdc = svd.matrixV().leftCols(k);
        Eigen::VectorXd S_bdc = svd.singularValues().head(k);

        Eigen::MatrixXd A_approx_bdc = U_bdc * S_bdc.asDiagonal() * V_bdc.transpose();
        double bdc_error = (A_bdc - A_approx_bdc).norm() / A_norm;

        // BDCSVD 作为基准，速度比为 1.0
        csv_file << "BDCSVD," << k << ","
                 << bdc_time << ","
                 << 1.0 << ","
                 << bdc_error << "\n";

        // ===========================================================
        // 3.2) Power Method SVD
        // ===========================================================
        {
            start = std::chrono::high_resolution_clock::now();

            Eigen::PowerMethodSVD<Eigen::SparseMatrix<double>> pmsvd;
            pmsvd.compute(A, k);

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_pm = end - start;
            double pm_time = elapsed_pm.count();

            Eigen::MatrixXd U_pm = pmsvd.matrixU();
            Eigen::VectorXd S_pm = pmsvd.singularValues();
            Eigen::MatrixXd V_pm = pmsvd.matrixV();

            Eigen::MatrixXd A_approx_pm = U_pm * S_pm.asDiagonal() * V_pm.transpose();
            double pm_error = (A_bdc - A_approx_pm).norm() / A_norm;

            double pm_speedup = bdc_time / pm_time;
            csv_file << "PowerMethodSVD," << k << ","
                     << pm_time << ","
                     << pm_speedup << ","
                     << pm_error << "\n";
        }

        // ===========================================================
        // 3.3) Randomized SVD (1 power iteration)
        // ===========================================================
        {
            start = std::chrono::high_resolution_clock::now();

            // 同理，若 RandomizedSVD 是模板实现且支持 SparseMatrix：
            Eigen::RandomizedSVD<Eigen::SparseMatrix<double>> rsvd_iter1;
            rsvd_iter1.compute(A, k, 1);

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_rsvd1 = end - start;
            double rsvd1_time = elapsed_rsvd1.count();

            Eigen::MatrixXd A_approx_rsvd1 =
                    rsvd_iter1.matrixU() *
                    rsvd_iter1.singularValues().asDiagonal() *
                    rsvd_iter1.matrixV().transpose();
            double rsvd1_error = (A_bdc - A_approx_rsvd1).norm() / A_norm;

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

            Eigen::RandomizedSVD<Eigen::SparseMatrix<double>> rsvd_iter2;
            rsvd_iter2.compute(A, k, 2);

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_rsvd2 = end - start;
            double rsvd2_time = elapsed_rsvd2.count();

            Eigen::MatrixXd A_approx_rsvd2 =
                    rsvd_iter2.matrixU() *
                    rsvd_iter2.singularValues().asDiagonal() *
                    rsvd_iter2.matrixV().transpose();
            double rsvd2_error = (A_bdc - A_approx_rsvd2).norm() / A_norm;

            double rsvd2_speedup = bdc_time / rsvd2_time;
            csv_file << "RandomizedSVD_Iter2," << k << ","
                     << rsvd2_time << ","
                     << rsvd2_speedup << ","
                     << rsvd2_error << "\n";
        }

        std::cout << "Done for k = " << k << "\n" << std::endl;
    }

    csv_file.close();
    std::cout << "Results saved to svd_results_sparse.csv" << std::endl;

    return 0;
}