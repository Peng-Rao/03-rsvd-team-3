//
// Created by 黄家励 on 2025/1/27.
//
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "RandomizedSVD.h"
#include "PCA.h"

using namespace std;

Eigen::MatrixXd PCA(const Eigen::MatrixXd& data, int num_components, int rank) {
    // Step 1: data centralisation
    Eigen::RowVectorXd mean = data.colwise().mean();  // get mean value of each col
    Eigen::MatrixXd centered = data.rowwise() - mean;

    // Step 2: Calculate the eigen value and eigen vector by rSVD instead of calculating covariance matrix

    // Create an instance of RandomizedSVD
    Eigen::RandomizedSVD<Eigen::MatrixXd> rSVD;

    // Compute the SVD for the current data matrix
    rSVD.compute(centered, rank);
    Eigen::VectorXd eigen_values = rSVD.singularValues();       // 特征值
    Eigen::MatrixXd eigen_vectors = rSVD.matrixV();     // 特征向量

    // Step 4: ordering by eigen value
    vector<pair<double, Eigen::VectorXd>> eigen_pairs;
    for (int i = 0; i < eigen_values.size(); ++i) {
        eigen_pairs.emplace_back(eigen_values(i), eigen_vectors.col(i));
    }
    sort(eigen_pairs.rbegin(), eigen_pairs.rend(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // extract num principal components
    Eigen::MatrixXd projection_matrix(data.cols(), num_components);
    for (std::vector<std::pair<double, Eigen::VectorXd>>::size_type i = 0; i < num_components; ++i) {
        projection_matrix.col(i) = eigen_pairs[i].second;
    }

    // Step 5: data projection
    return centered * projection_matrix;
}
