add_library(pca_lib
    PCA.h
)

target_include_directories(pca_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pca_lib PUBLIC
    Eigen3::Eigen
)

# Make this a header-only library
set_target_properties(pca_lib PROPERTIES LINKER_LANGUAGE CXX) 