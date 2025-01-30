# config for unit tests
add_executable(RSVDTest test_rsvd.cpp)
target_link_libraries(RSVDTest PRIVATE Catch2::Catch2WithMain)
target_link_libraries(RSVDTest PRIVATE Eigen3::Eigen)
target_link_libraries(RSVDTest PRIVATE OpenMP::OpenMP_CXX)

add_executable(GivensRotationQRTest test_givens_rotation_qr.cpp)
target_link_libraries(GivensRotationQRTest PRIVATE Catch2::Catch2WithMain)
target_link_libraries(GivensRotationQRTest PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)

# Add PCA test
add_executable(PCATest test_pca.cpp)
target_link_libraries(PCATest PRIVATE 
    Catch2::Catch2WithMain
    Eigen3::Eigen
    pca_lib
)
target_include_directories(PCATest PRIVATE 
    ${CMAKE_SOURCE_DIR}/applications
)

# Add Digits Recognition test
add_executable(DigitsRecognitionTest test_digits_recognition.cpp)
target_link_libraries(DigitsRecognitionTest PRIVATE 
    Catch2::Catch2WithMain
    Eigen3::Eigen
    ${OpenCV_LIBS}
    pca_lib
    digits_recognition_lib
)
target_include_directories(DigitsRecognitionTest PRIVATE 
    ${CMAKE_SOURCE_DIR}/applications
    ${CMAKE_SOURCE_DIR}/applications/digits
)
target_compile_definitions(DigitsRecognitionTest PRIVATE 
    MNIST_DATA_DIR="${MNIST_DATA_DIR}"
)


