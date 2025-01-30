add_library(digits_recognition_lib
    digits_recognition.cpp
    digits_recognition.hpp
)

target_link_libraries(digits_recognition_lib PUBLIC
    pca_lib
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_include_directories(digits_recognition_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/applications
) 