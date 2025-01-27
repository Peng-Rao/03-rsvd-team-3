// Uncomment the following line if you want to use BLAS with Eigen
// #define EIGEN_USE_BLAS

#include "PowerMethodSVD.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <random>

// Function to generate a random sparse matrix with a given sparsity
Eigen::SparseMatrix<double> generateRandomSparseMatrix(int rows, int cols, double sparsity, std::mt19937 &gen) {
    Eigen::SparseMatrix<double> mat(rows, cols);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> val_dis(-1.0, 1.0);

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(static_cast<int>(rows * cols * sparsity));

    for(int k = 0; k < cols; ++k){
        for(int i = 0; i < rows; ++i){
            if(dis(gen) < sparsity){
                tripletList.emplace_back(i, k, val_dis(gen));
            }
        }
    }

    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    mat.makeCompressed();
    return mat;
}

int main() {
    // Seed for reproducibility
    std::mt19937 gen(42);

    // Define matrix dimensions and rank
    const int rows = 100;
    const int cols = 100;
    const int low_rank = 50;
    const double sparsity = 0.05; // 5% non-zero entries

    // Generate sparse random matrices U and V
    Eigen::SparseMatrix<double> U = generateRandomSparseMatrix(rows, low_rank, sparsity, gen);
    Eigen::SparseMatrix<double> V = generateRandomSparseMatrix(cols, low_rank, sparsity, gen);

    // Construct the low-rank sparse matrix A = U * V^T
    Eigen::SparseMatrix<double> A = U * V.transpose();

    // Optionally, you can make A explicitly low-rank by keeping only the top `low_rank` singular values
    // However, since A is constructed as U * V^T with U and V having rank `low_rank`, A should inherently have rank ≤ low_rank

    // Define the number of singular values/components to compute
    int k = 50;

    // Perform Randomized SVD using PowerMethodSVD
    // Ensure that PowerMethodSV is templated to accept Eigen::SparseMatrix<double>
    Eigen::PowerMethodSVD<Eigen::SparseMatrix<double>> pmsvd;
    pmsvd.compute(A, k, 1e-9, 0.05, 0.001);

    // Retrieve the factors from the SVD
    Eigen::MatrixXd U_svd = pmsvd.matrixU();
    Eigen::VectorXd singularValues = pmsvd.singularValues();
    Eigen::MatrixXd V_svd = pmsvd.matrixV();

    // To compute the approximation A_approx = U * S * V^T, we need to handle sparse and dense matrices appropriately
    Eigen::MatrixXd U_dense = Eigen::MatrixXd(U_svd);
    Eigen::MatrixXd V_dense = Eigen::MatrixXd(V_svd);
    Eigen::MatrixXd S = singularValues.asDiagonal();

    Eigen::MatrixXd A_approx = U_dense * S * V_dense.transpose();

    // Compute the Frobenius norm of the approximation error
    // Convert A to a dense matrix for comparison
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    double frobenius_norm = (A_dense - A_approx).norm() / A_dense.norm();

    std::cout << "Frobenius norm of approximation error: " << frobenius_norm << std::endl;

    return 0;
}
