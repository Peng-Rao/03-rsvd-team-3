#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>

constexpr int n{100};
constexpr int rank{10};
constexpr int powerIterations{2};

Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

TEST_CASE("RSVD Benchmark on randomized matrix[omp]", "[rsvd_bench_hilbert]") {
    BENCHMARK("RSVD compute") {
        rsvd.compute(A, rank, powerIterations);
    };
}