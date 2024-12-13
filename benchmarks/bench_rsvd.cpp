#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <thread>
#include <vector>
#include <string>

constexpr int n{2000};
constexpr int rank{10};
constexpr int powerIterations{2};

Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

void setEigenThreads(int numThreads) {
    Eigen::setNbThreads(numThreads);
}

TEST_CASE("RSVD Benchmark with multi-thread", "[rsvd_bench]") {
    int hwCores = static_cast<int>(std::thread::hardware_concurrency());
    if (hwCores == 0) {
        hwCores = 8;
    }
    int maxThreads = hwCores * 2;

    std::vector threadCounts = { maxThreads, maxThreads / 2, maxThreads / 4, 1 };

    for (auto t : threadCounts) {
        BENCHMARK("RSVD compute with multi-thread(" + std::to_string(t) + ")") {
            setEigenThreads(t);
            rsvd.compute(A, rank, powerIterations);
        };
    }
}