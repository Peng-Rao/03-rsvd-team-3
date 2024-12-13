// #define EIGEN_USE_BLAS

#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

constexpr int n{2000};
constexpr int rank{10};
constexpr int powerIterations{2};

Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);

Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

void setEigenThreads(int numThreads) {
    Eigen::setNbThreads(numThreads);
}

TEST_CASE("RSVD Benchmark with multi-thread", "[rsvd_bench]") {
    // int defaultThreads = Eigen::nbThreads();
    
    BENCHMARK("RSVD compute with multi-thread(64)") {
        setEigenThreads(64);
        rsvd.compute(A, rank, powerIterations);
    };
    
    BENCHMARK("RSVD compute with multi-thread(32)") {
        setEigenThreads(32);
        rsvd.compute(A, rank, powerIterations);
    };

    BENCHMARK("RSVD compute with multi-thread(16)") {
        setEigenThreads(16);
        rsvd.compute(A, rank, powerIterations);
    };

    BENCHMARK("RSVD compute with multi-thread(8)") {
        setEigenThreads(8);
        rsvd.compute(A, rank, powerIterations);
    };

    BENCHMARK("RSVD compute with multi-thread(4)") {
        setEigenThreads(4);
        rsvd.compute(A, rank, powerIterations);
    };

    BENCHMARK("RSVD compute with multi-thread(1)") {
        setEigenThreads(1);
        rsvd.compute(A, rank, powerIterations);
    };
}