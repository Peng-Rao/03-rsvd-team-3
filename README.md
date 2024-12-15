# Randomized SVD
- [Overview](#overview)
- [Algorithms description](#algorithms-description)
  + [PowerMethod SVD](#powermethod-svd)
- [Project setup](#project-setup)
  + [MacOS setup](#macos-setup)
  + [Windows setup](#windows-setup)
  + [Compiler setup](#compiler-setup)
  + [Build the project](#build-the-project)
- [RandomizedSVD.h code explanation](#randomizedsvdh-code-explanation)

## Overview
This project is a C++ implementation of the **Randomized Singular Value Decomposition (rSVD)** algorithm. We only used the matrix operations of the `Eigen` library to implement our algorithm. We do some benchmarks to compare the performance of our implementation with the `Eigen` library, the result indicates that our implementation can enhance the performance of handling large or sparse matrices.

## Algorithms description

### PowerMethod SVD
The `PowerMethodSVD` class is a template-based implementation for computing the Singular Value Decomposition (SVD) of a matrix (dense or sparse) using the **Power Method with Deflation**. This approach is designed for approximating a few dominant singular values and their associated singular vectors efficiently.

In order to make the operation more efficient, we have optimized the algorithm. Alternately apply matrix A and its transpose $A^T$, thereby implicitly performing the power method on $A^T A$ without explicitly constructing $A^T A$.

To compute the largest singular value  $\sigma_1$  and its corresponding singular vectors  $u_1$  and  $v_1$ , perform the following steps:
1. Initialize $v_1$ as a random vector:
$$v_1 \leftarrow \text{random vector of size } n$$
2. Normalize $v_1$ :
$$v_1 \leftarrow \frac{v_1}{\|v_1\|}$$
3. Iteratively compute:
   - Left singular vector:
     $$u_1 \leftarrow \frac{A v_1}{\|A v_1\|}$$
   - Right singular vector:
     $$v_1 \leftarrow \frac{A^\top u_1}{\|A^\top u_1\|}$$
   - Convergence is checked by monitoring the change in $v_1$:
     $$\|v_1^{(t+1)} - v_1^{(t)}\| < \text{tol}$$
4. After convergence, the largest singular value is:
   $$\sigma_1 = \|A v_1\|$$
5. Normalize $u_1$ to ensure consistency:
   $$u_1 = \frac{A v_1}{\sigma_1}$$

 After computing $\sigma_1, u_1, v_1$, deflate $A$ to remove the contribution of the largest singular component:
 $$A \leftarrow A - \sigma_1 u_1 v_1^\top$$

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

### Windows setup

For Windows, see this guide: https://learn.microsoft.com/en-us/vcpkg/get_started/get-started/

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
We have implemented a benchmark to compare the performance of our implementation. The benchmark is run on a dense matrix of size 1000x1000 and a sparse matrix of size 1000x1000. The benchmark measures the time taken to compute the SVD of the matrix using our implementation and the `Eigen` library. The results show that our implementation is faster than the `Eigen` library for both dense and sparse matrices.

| Benchmark Name                        | Size  | Mean          | Low Mean      | High Mean     |
|---------------------------------------|-------|---------------|---------------|---------------|
| GivensRotationQR with sparse matrix   | 100   | 2.14231 ms    | 2.13479 ms    | 2.15039 ms    |
| GivensRotationQR with sparse matrix   | 200   | 11.9559 ms    | 11.9181 ms    | 11.9883 ms    |
| GivensRotationQR with sparse matrix   | 300   | 38.4835 ms    | 38.3569 ms    | 38.6319 ms    |
| GivensRotationQR with sparse matrix   | 400   | 86.3512 ms    | 86.2812 ms    | 86.4386 ms    |
| GivensRotationQR with sparse matrix   | 500   | 162.965 ms    | 161.811 ms    | 166.101 ms    |
| GivensRotationQR with sparse matrix   | 600   | 279.566 ms    | 278.516 ms    | 281.161 ms    |
| GivensRotationQR with sparse matrix   | 700   | 435.05 ms     | 433.854 ms    | 436.587 ms    |
# RandomizedSVD.h code explanation
The Randomized Singular Value Decomposition is an algorithm for efficient approximation of the SVD of large matrices. It is particularly effective when the desired decomposition rank is smaller than the input matrix dimensions. 
It is written using the Eigen namespace. 
The RandomizedSVD template class has two parameters:
  - The type of the input matrix
  - The decomposition options
It uses the default constructor.
It has 4 public methods:
 - compute(): contains the rSVD algorithm
 - singularValue(): returns the vector of singular values
 - matrixU(): returns the left singular vectors
 - matrixV(): returns the right singular vectors
Then there are 2 private methods:
 - generateRandomMatrix(): generates a 2D random matrix with random Gaussian values      given the matrix dimensions.
 - randomProjection(): this methods is the core of the rSVD decomposition. It generates a random Gaussian matrix and multiplies the input matrix by it, creating a "sketch" matrix. Then it performs the number of Power Iteraions requested on the sketc to refine it. Using the HouseholderQR function, it computer the QR decomposition of the sketch. Finally, it projects the original matrix onto the low-dimensional subspace and performs the SVD on it using the JacobiSVD function.