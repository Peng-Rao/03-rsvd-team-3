# Randomized SVD
- [Overview](#overview)
- [Programming Paradigm](#programming-paradigm)
- [Project setup](#project-setup)
  + [MacOS setup](#macos-setup)
  + [Windows WSL2 setup](#windows-wsl2-setup)
  + [Compiler setup](#compiler-setup)
  + [Build the project](#build-the-project)
- [Benchmarks](#benchmarks)
- [Profiling](#profiling)

## Overview
This project is a C++ implementation of the **Randomized Singular Value Decomposition (rSVD)** algorithm. We used the matrix operations of the `Eigen` library to implement our algorithm. We do some benchmarks to compare the performance of our implementation with the `Eigen` library, the result indicates that our implementation can enhance the performance of handling large or sparse matrices.

## Programming Paradigm
Our project leverages template programming to provide flexible and efficient implementations of key linear algebra algorithms. Specifically, the classes `GivensRotationQR`, `PowerMethodSVD`, and `RandomizedSVD` are designed to accommodate a wide range of data types and storage formats. These classes support both dense and sparse matrices. For example:
```cpp
// Dense matrix
Eigen::RandomizedSVD<double, Eigen::Dynamic, Eigen::Dynamic> rsvd;

// Sparse matrix
Eigen::RandomizedSVD<Eigen::SparseMatrix<double>>

// Row-major matrix
Eigen::RandomizedSVD<Eigen::SparseMatrix<double, Eigen::RowMajor>>
```


## Project setup
We use `CMake` to build the project, and `vcpkg` to manage dependencies, our project can run across platforms.

Prerequisites:
- [CMake](https://cmake.org/download/)
- [vcpkg](https://github.com/microsoft/vcpkg)
- [C++ compiler](https://code.visualstudio.com/docs/languages/cpp#_install-a-compiler)

### MacOS setup
For `MacOS`, install `CMake` and `Vcpkg`. If you need `OpenMP` support, you must install `llvm`:
```bash
brew install cmake
brew install vcpkg
brew install llvm
```
To use `vcpkg`:
```bash
git clone https://github.com/microsoft/vcpkg "$HOME/vcpkg"
export VCPKG_ROOT="$HOME/vcpkg
$VCPKG_ROOT/bootstrap-vcpkg.sh
```
After installing the above packages, you need to load them into `PATH`, for `MacOS`, edit the `.zshrc` file:
```bash
# Add vcpkg to PATH
export VCPKG_ROOT=... # your vcpkg path
export PATH=$VCPKG_ROOT:$PATH

# Add llvm to PATH
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/libomp/bin"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```
And then `MacOS` environment settings completed.

### Windows WSL2 setup


1. install CMake and Vcpkg
```bash
sudo apt update
sudo apt install cmake
git clone https://github.com/microsoft/vcpkg "$HOME/vcpkg"
$HOME/vcpkg/bootstrap-vcpkg.sh
```
2. set vcpkg to PATH
```bash
echo 'export VCPKG_ROOT=$HOME/vcpkg' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Fork** and **Clone** this project to your own repo.

### Compiler setup
For `MacOS`, you can use `clang` or `gcc`, for `Windows`, you can use `MSVC` or `gcc`. First you should add `CMakeUserPresets.json` to the project root directory, for example I have configured the `CMakeUserPresets.json` file as follows, you should replace the compiler path with your own path, and the `VCPKG_ROOT` with your own path:
```json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "gcc",
      "inherits": "vcpkg",
      "environment": {
        "VCPKG_ROOT": "/Users/raopend/vcpkg"
      },
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
        "CMAKE_C_COMPILER": "/opt/homebrew/bin/gcc-14",
        "CMAKE_CXX_COMPILER": "/opt/homebrew/bin/g++-14"
      }
    },
    {
      "name": "clang",
      "inherits": "vcpkg",
      "environment": {
        "VCPKG_ROOT": "/Users/raopend/vcpkg"
      },
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
        "CMAKE_C_COMPILER": "/opt/homebrew/opt/llvm/bin/clang",
        "CMAKE_CXX_COMPILER": "/opt/homebrew/opt/llvm/bin/clang++"
      }
    }
  ]
}
```
You can replace the compiler path with your own path.

### Build the project
After setting up the environment, you can build the project. The keyword `default` is the preset name in the `CMakeUserPresets.json` file, for example, I use the `clang` preset:
1.  Configure the build using CMake:
```bash 
cmake --preset=clang
```
2. Build the project
```bash
cmake --build build
```
3. Run the application
```bash
./build/benchmarks/BenchRSVD
```

# Benchmarks
To run the Benchmarks, you can use the following command:
```bash
./build/benchmarks/dense/dense_svd # Dense matrix SVD
./build/benchmarks/sparse/sparse_svd # Sparse matrix SVD
./build/benchmarks/sparse/sparse_qr # Sparse matrix QR
./build/benchmarks/sparse/givens_qr # Givens Rotation QR
```

# Profiling
To profile the application, you can use the following command(Only for MacOS):
```bash
xcrun xctrace record --template 'Time Profiler' --launch ./rsvd_dense_profiling
```
And then you can open the `rsvd_dense_profiling.trace` file to view the profiling results.

# Image compression
See the [`README.md`](./applications/imagecompression/README.md) in the `image_compression` directory for more information.

# CUR decomposition
See the [`README.md`](./src/README.md) in the `CUR_decomposition` directory for more information.