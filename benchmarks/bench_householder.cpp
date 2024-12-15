#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/SparseQR>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <vector>
#include <string>
#include <random>

std::vector<Eigen::MatrixXd> generateDenseMatrices(const int startSize, const int endSize, const int step) {
    std::vector<Eigen::MatrixXd> denseMatrices;
    for(int size = startSize; size <= endSize; size += step) {
        denseMatrices.emplace_back(Eigen::MatrixXd::Random(size, size));
    }
    return denseMatrices;
}

std::vector<Eigen::SparseMatrix<double>> generateSparseMatrices(const int startSize, const int endSize,
                                                                                 const int step, const double sparsity) {
    std::vector<Eigen::SparseMatrix<double>> sparseMatrices;
    sparseMatrices.reserve((endSize - startSize) / step + 1);

    std::mt19937 gen(42);
    std::uniform_real_distribution dis(0.0, 1.0);

    for(int size = startSize; size <= endSize; size += step) {
        Eigen::SparseMatrix<double> sparse(size, size);
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(static_cast<int>(size * size * sparsity));

        for(int i = 0; i < size; ++i) {
            for(int j = 0; j < size; ++j) {
                if(dis(gen) < sparsity) {
                    tripletList.emplace_back(i, j, dis(gen));
                }
            }
        }
        sparse.setFromTriplets(tripletList.begin(), tripletList.end());
        sparseMatrices.emplace_back(std::move(sparse));
    }
    return sparseMatrices;
}

constexpr int startSize = 100;
constexpr int endSize = 1000;
constexpr int stepSize = 100;

constexpr double sparsity = 0.1;
std::mt19937 gen(42);
std::uniform_real_distribution dis(0.0, 1.0);
auto denseMatrices = generateDenseMatrices(startSize, endSize, stepSize);
auto sparseMatrices = generateSparseMatrices(startSize, endSize, stepSize, sparsity);


TEST_CASE("Eigen Householder Decomposition Benchmark", "[householder_bench_dynamic]") {
    for(size_t idx = 0; idx < denseMatrices.size(); ++idx) {
        const int size = startSize + static_cast<int>(idx) * stepSize;
        const auto& dense = denseMatrices[idx];
        auto& sparse = sparseMatrices[idx];

        BENCHMARK("HouseholderQR with dense matrix size " + std::to_string(size)) {
            Eigen::HouseholderQR<Eigen::MatrixXd> householderQR;
            householderQR.compute(dense);
        };

        BENCHMARK("SparseQR with sparse matrix size " + std::to_string(size)) {
            sparse.makeCompressed();
            Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> sparseQR;
            sparseQR.compute(sparse);
        };
    }
}