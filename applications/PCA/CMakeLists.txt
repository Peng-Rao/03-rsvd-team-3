add_library(pca_lib PCA.cpp)
target_link_libraries(pca_lib PUBLIC Eigen3::Eigen)
target_include_directories(pca_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Remove or comment out the executable target if it exists
# add_executable(pca_main PCA.cpp)
