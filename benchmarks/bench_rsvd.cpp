#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

constexpr int n{500};
constexpr int rank{10};
constexpr int powerIterations{2};

Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);

Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

void setEigenThreads(int numThreads) {
    Eigen::setNbThreads(numThreads);
}

TEST_CASE("RSVD Benchmark with and without OpenMP", "[rsvd_bench]") {
    int defaultThreads = Eigen::nbThreads();

    BENCHMARK("RSVD compute with OpenMP") {
        setEigenThreads(defaultThreads);
        rsvd.compute(A, rank, powerIterations);
    };

    BENCHMARK("RSVD compute without OpenMP") {
        setEigenThreads(1);
        rsvd.compute(A, rank, powerIterations);
    };
}