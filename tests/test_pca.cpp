#include <catch2/catch_test_macros.hpp>
#include "PCA.h"

TEST_CASE("PCA Basic Test", "[pca]") {
    // Test data
    Eigen::MatrixXd data(5, 3);
    data << 2.5, 2.4, 1.5,
            0.5, 0.7, 0.3,
            2.2, 2.9, 1.6,
            1.9, 2.2, 1.1,
            3.1, 3.0, 1.8;

    SECTION("Dimension Reduction") {
        Eigen::MatrixXd reduced_data = PCA(data, 2, 2);
        
        // Check dimensions
        REQUIRE(reduced_data.rows() == 5);
        REQUIRE(reduced_data.cols() == 2);
        
        // Basic sanity checks
        REQUIRE(reduced_data.allFinite());
    }
} 