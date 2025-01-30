#include <catch2/catch_test_macros.hpp>
#include "digits/digits_recognition.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("MNIST Dataset Test", "[digits][mnist]") {
    std::cout << "================================MNIST Dataset Test================================================" << std::endl;
    std::string data_dir = MNIST_DATA_DIR;
    std::string train_images_path = data_dir + "/train-images-idx3-ubyte";
    std::string train_labels_path = data_dir + "/train-labels-idx1-ubyte";
    std::string test_images_path = data_dir + "/t10k-images-idx3-ubyte";
    std::string test_labels_path = data_dir + "/t10k-labels-idx1-ubyte";
    

    // Load training data - reduced dataset size
    auto [training_data, training_labels] = MNISTLoader::loadMNISTData(
        train_images_path, 
        train_labels_path,
        10000  
    );
    
    // Load test data
    auto [test_data, test_labels] = MNISTLoader::loadMNISTData(
        test_images_path,
        test_labels_path,
        1000  
    );

    std::vector<int> components = {2,4,8,16,32,64,128,256};
    
    for(int num_components : components) {
        SECTION("Load and Train with " + std::to_string(num_components) + " components") {
            // Create and train model
            DigitsRecognition recognition(num_components);  
            recognition.train(training_data, training_labels);
            
            // Evaluate
            double accuracy = recognition.evaluate(test_data, test_labels);
            std::cout << "Accuracy with " << num_components << " components: " << accuracy << std::endl;
        }
    }
}

TEST_CASE("Single Image Prediction Test", "[digits][predict]") {
    std::cout << "================================Single Image Prediction Test================================================" << std::endl;
    std::string data_dir = MNIST_DATA_DIR;
    std::string train_images_path = data_dir + "/train-images-idx3-ubyte";
    std::string train_labels_path = data_dir + "/train-labels-idx1-ubyte";
    std::string test_image_path = data_dir + "/test_digit6.bmp";
    std::cout << "test_image_path: " << test_image_path << std::endl;
    
    SECTION("Load, Train and Predict") {
        // Load training data
        auto [training_data, training_labels] = MNISTLoader::loadMNISTData(
            train_images_path, 
            train_labels_path,
            1000  // Use 1000 training samples
        );
        
        // Create and train model
        DigitsRecognition recognition(50);
        recognition.train(training_data, training_labels);
        
        // Load and preprocess test image
        cv::Mat image = cv::imread(test_image_path, cv::IMREAD_GRAYSCALE);
        if(image.empty()) {
            std::cout << "Failed to load image" << std::endl;
            REQUIRE(false);  // Make test fail if loading fails
        } else {
            std::cout << "Image loaded successfully. Size: " << image.size() << std::endl;
        }
        
        // Resize to 28x28 if needed
        if (image.size() != cv::Size(28, 28)) {
            cv::resize(image, image, cv::Size(28, 28));
        }
        
        // Convert to Eigen vector
        Eigen::VectorXd test_vector = mat_to_eigen(image);
        
        // Predict
        int predicted_digit = recognition.predict(test_vector);
        std::cout << "Predicted digit: " << predicted_digit << std::endl;
        
        // Since we know this is digit 6
        REQUIRE(predicted_digit == 6);
    }
} 