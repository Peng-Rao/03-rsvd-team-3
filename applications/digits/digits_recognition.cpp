#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <RandomizedSVD.h>
#include "digits_recognition.hpp"

namespace fs = std::filesystem;

// MNISTLoader implementation
std::pair<Eigen::MatrixXd, std::vector<int>> MNISTLoader::loadMNISTData(
        const std::string& images_path,
        const std::string& labels_path,
        int max_images) {
    // Read images
    std::ifstream file(images_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + images_path);
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    if (max_images != -1 && max_images < num_images) {
        num_images = max_images;
    }

    // Read image data
    Eigen::MatrixXd images(num_images, num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images(i, j) = static_cast<double>(pixel) / 255.0; // Normalize to [0,1]
        }
    }

    // Read labels
    std::ifstream label_file(labels_path, std::ios::binary);
    if (!label_file.is_open()) {
        throw std::runtime_error("Cannot open file: " + labels_path);
    }

    int label_magic_number = 0, num_labels = 0;
    label_file.read((char*)&label_magic_number, sizeof(label_magic_number));
    label_file.read((char*)&num_labels, sizeof(num_labels));

    label_magic_number = __builtin_bswap32(label_magic_number);
    num_labels = __builtin_bswap32(num_labels);

    std::vector<int> labels(num_images);
    for (int i = 0; i < num_images; ++i) {
        unsigned char label = 0;
        label_file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return {images, labels};
}

// Helper function implementation
Eigen::VectorXd mat_to_eigen(const cv::Mat& image) {
    cv::Mat flat = image.reshape(1, 1);
    Eigen::VectorXd eigen_image(flat.cols);
    for(int i = 0; i < flat.cols; ++i) {
        eigen_image(i) = flat.at<uchar>(0, i) / 255.0;
    }
    return eigen_image;
}

// Constructor implementation
DigitsRecognition::DigitsRecognition(int num_components) : num_components_(num_components) {}

// Train implementation
void DigitsRecognition::train(const Eigen::MatrixXd& training_data, const std::vector<int>& labels) {
    labels_ = labels;
    
    // Store training data dimensions
    mean_ = training_data.colwise().mean();
    Eigen::MatrixXd centered = training_data.rowwise() - mean_;
    
    // Use PCA class from PCA library
    Eigen::PCA pca;
    pca.computeByRSVD(training_data, num_components_, num_components_);
    Eigen::MatrixXd reduced = pca.reducedMatrix();
    
    // Store the projection matrix for later use
    projection_matrix_ = (centered.transpose() * centered).ldlt().solve(centered.transpose() * reduced);
    
    // Store reduced training data
    reduced_training_data_.clear();
    for(int i = 0; i < reduced.rows(); ++i) {
        reduced_training_data_.push_back(reduced.row(i).transpose());
    }
    
    // Debug info
    std::cout << "Training data dimensions: " << training_data.rows() << "x" << training_data.cols() << std::endl;
    std::cout << "Mean dimensions: " << mean_.rows() << "x" << mean_.cols() << std::endl;
    std::cout << "Projection matrix dimensions: " << projection_matrix_.rows() << "x" << projection_matrix_.cols() << std::endl;
    std::cout << "Reduced data dimensions: " << reduced.rows() << "x" << reduced.cols() << std::endl;
}

// Predict implementation
int DigitsRecognition::predict(const Eigen::VectorXd& test_image) {
    // Convert test_image to row vector if needed
    Eigen::RowVectorXd test_row = (test_image.cols() == 1) ? test_image.transpose() : test_image;
    
    // Center and project the test image
    Eigen::RowVectorXd centered = test_row - mean_;
    Eigen::RowVectorXd reduced_test = centered * projection_matrix_;
    
    // Find nearest neighbor using reduced representations
    double min_distance = std::numeric_limits<double>::max();
    int predicted_label = -1;
    
    for(size_t i = 0; i < reduced_training_data_.size(); ++i) {
        // Convert reduced_training_data_[i] to row vector for consistent comparison
        Eigen::RowVectorXd train_row = reduced_training_data_[i].transpose();
        double distance = (train_row - reduced_test).squaredNorm();
        if(distance < min_distance) {
            min_distance = distance;
            predicted_label = labels_[i];
        }
    }
    
    return predicted_label;
}

// Evaluate implementation
double DigitsRecognition::evaluate(const Eigen::MatrixXd& test_data, const std::vector<int>& test_labels) {
    int correct = 0;
    int total = test_data.rows();
    
    for(int i = 0; i < total; ++i) {
        int predicted = predict(test_data.row(i));
        if(predicted == test_labels[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / total;
} 