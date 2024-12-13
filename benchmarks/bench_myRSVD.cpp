#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <iostream>

// Test different matrix sizes for better performance comparison
std::vector<int> sizes = {500};
constexpr int rank{10};
constexpr int powerIterations{2};

TEST_CASE("Custom RSVD Implementation Benchmark", "[custom_rsvd_bench]") {
    for (int size : sizes) {
        // Create test matrix for each size
        Eigen::MatrixXd testMatrix = Eigen::MatrixXd::Random(size, size);
        Eigen::RandomizedSVD<Eigen::MatrixXd> myRSVD;
        
        std::string sizeStr = std::to_string(size);
        
        // Test original implementation
        BENCHMARK(("Custom RSVD without OpenMP (size=" + sizeStr + ")").c_str()) {
            myRSVD.compute(testMatrix, rank, powerIterations);
        };

        // Add debug prints before OpenMP benchmark
        BENCHMARK(("Custom RSVD with OpenMP (size=" + sizeStr + ")").c_str()) {
            int defaultThreads = Eigen::nbThreads();
            std::cout<<"defaultThreads:"<< defaultThreads<<std::endl;
            Eigen::setNbThreads(4);
            myRSVD.compute(testMatrix, rank, powerIterations);
        };
    }

}
