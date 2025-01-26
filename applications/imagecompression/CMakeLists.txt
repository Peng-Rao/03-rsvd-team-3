find_package(Stb REQUIRED)

add_executable(ImageCompression ImageCompression.cpp)
target_link_libraries(ImageCompression PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(ImageCompression PRIVATE ${Stb_INCLUDE_DIR})
target_compile_options(ImageCompression PRIVATE -O3 -march=native)