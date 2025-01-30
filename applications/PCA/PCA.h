#pragma once
#include <Eigen/Dense>

Eigen::MatrixXd PCA(const Eigen::MatrixXd& data, int num_components, int rank); 