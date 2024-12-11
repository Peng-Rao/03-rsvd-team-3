#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

const int rows = 1000;
const int cols = 1000;

Eigen::MatrixXd A = Eigen::MatrixXd::Random(rows, cols);
Eigen::MatrixXd B = Eigen::MatrixXd::Random(cols, rows);

void setEigenThreads(int numThreads) {
    Eigen::setNbThreads(numThreads);
}

TEST_CASE("Matrix Multiplication Benchmark with and without OpenMP", "[matmul_bench]") {
    int defaultThreads = Eigen::nbThreads();

    BENCHMARK("Matrix Multiplication with OpenMP") {
        setEigenThreads(defaultThreads);
        Eigen::MatrixXd C = A * B;
    };

    BENCHMARK("Matrix Multiplication without OpenMP") {
        setEigenThreads(1);
        Eigen::MatrixXd C = A * B;
    };
}