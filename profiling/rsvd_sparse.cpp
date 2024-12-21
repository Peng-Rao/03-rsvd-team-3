#include "RandomizedSVD.h"

#include <Eigen/Sparse>

std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> generateSparseMatrices(const int startSize, const int endSize,
                                                                                 const int step, const double sparsity) {
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> sparseMatrices;
    sparseMatrices.reserve((endSize - startSize) / step + 1);

    std::mt19937 gen(42);
    std::uniform_real_distribution dis(0.0, 1.0);

    for (int size = startSize; size <= endSize; size += step) {
        Eigen::SparseMatrix<double, Eigen::RowMajor> sparse(size, size);
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(static_cast<int>(size * size * sparsity));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (dis(gen) < sparsity) {
                    tripletList.emplace_back(i, j, dis(gen));
                }
            }
        }
        sparse.setFromTriplets(tripletList.begin(), tripletList.end());
        sparseMatrices.emplace_back(std::move(sparse));
    }
    return sparseMatrices;
}



int main() {
    // Global constants for matrix generation
    constexpr int startSize = 1000;
    constexpr int endSize = 1000;
    constexpr int stepSize = 1000;
    constexpr double sparsity = 0.1;

    // Generate matrices once to avoid recreating in each test case
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> sparseMatrices = generateSparseMatrices(startSize, endSize, stepSize, sparsity);
    constexpr int rank = 10;
    const auto& sparse = sparseMatrices[0];
    Eigen::RandomizedSVD<Eigen::SparseMatrix<double, Eigen::RowMajor>> rsvd;
    rsvd.compute(sparse, rank, 2);

    return 0;
}
