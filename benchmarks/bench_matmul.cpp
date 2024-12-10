#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

const int rows = 1000;
const int cols = 1000;

Eigen::MatrixXd A = Eigen::MatrixXd::Random(rows, cols);
Eigen::MatrixXd B = Eigen::MatrixXd::Random(cols, rows);

TEST_CASE("matmul", "[benchmark]") {
    BENCHMARK("matmul") {
        Eigen::MatrixXd C = A * B;
    };
}