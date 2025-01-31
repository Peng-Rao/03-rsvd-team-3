#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <PCA/PCA.h>
#include <vector>
#include <string>
#include <utility>

// Helper function declaration
Eigen::VectorXd mat_to_eigen(const cv::Mat& image);

class MNISTLoader {
public:
    static std::pair<Eigen::MatrixXd, std::vector<int>> loadMNISTData(
            const std::string& images_path,
            const std::string& labels_path,
            int max_images = -1);
};

class DigitsRecognition {
public:
    DigitsRecognition(int num_components = 50);
    void train(const Eigen::MatrixXd& training_data, const std::vector<int>& labels);
    int predict(const Eigen::VectorXd& test_image);
    double evaluate(const Eigen::MatrixXd& test_data, const std::vector<int>& test_labels);

private:
    Eigen::MatrixXd projection_matrix_;
    Eigen::RowVectorXd mean_;
    std::vector<Eigen::VectorXd> reduced_training_data_;
    std::vector<int> labels_;
    int num_components_;
}; 