# Randomized SVD
- [Overview](#overview)
- [Algorithms description](#algorithms-description)
  + [PowerMethod SVD](#powermethod-svd)
  + [Givens Rotation QR Decomposition](#givens-rotation-qr-decomposition)
- [Project setup](#project-setup)
  + [MacOS setup](#macos-setup)
  + [Windows WSL2 setup](#windows-wsl2-setup)
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

$$
v_1 \leftarrow \text{random vector of size } n
$$

2. Normalize $v_1$ :

$$
v_1 \leftarrow \frac{v_1}{\|v_1\|}
$$

3. Iteratively compute:
- Left singular vector:
   
$$
u_1 \leftarrow \frac{A v_1}{\|A v_1\|}
$$

- Right singular vector:
   
$$
v_1 \leftarrow \frac{A^\top u_1}{\|A^\top u_1\|}
$$

- Convergence is checked by monitoring the change in $v_1$:
   
$$
\|v_1^{(t+1)} - v_1^{(t)}\| < \text{tol}
$$

4. After convergence, the largest singular value is:
   
$$
\sigma_1 = \|A v_1\|
$$

5. The left singular vector is computed as:

$$
u_1 = \frac{A v_1}{\sigma_1}
$$

After computing $\sigma_1, u_1, v_1$, deflate $A$ to remove the contribution of the largest singular component:

$$
A \leftarrow A - \sigma_1 u_1 v_1^\top
$$

### Givens Rotation QR Decomposition
The `GivensRotationQR` class is a template-based implementation for computing the QR decomposition of a matrix using the **Givens Rotation** method. This approach is designed for sparse matrices, because of generics, it can be used for both dense and sparse matrices.

The **Givens Rotation QR** decomposition is a method for decomposing a matrix $A$ into an orthogonal matrix $Q$ and an upper triangular matrix $R$, such that:

$$
A = Q \cdot R
$$

A Givens rotation matrix is used to zero out specific elements of a matrix. For two elements $a$ and $b$, the Givens rotation coefficients $c$ and $s$ are computed as:

$$
r = \sqrt{a^2 + b^2}, \quad c = \frac{a}{r}, \quad s = -\frac{b}{r}
$$

The Givens rotation matrix for rows $i$ and $k$ is:

$$
G(i, k) =
\begin{bmatrix}
1 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots \\
0 & \cdots & c & \cdots & -s & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
0 & \cdots & s & \cdots & c & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & \cdots & 0 & \cdots & 1
\end{bmatrix}
$$

The steps for the Givens Rotation QR decomposition are:
1. Initialize $R$ as a copy of the input matrix $A$ and $Q$ as the identity matrix.
2. For each column $j$:
    - Apply Givens rotations to zero out elements below the diagonal in column $j$.
    - Update $R$ by applying Givens rotations from the left.
    - Accumulate rotations into $Q$ by applying Givens rotations from the right (transpose of $G$).
3. Return $Q$ and $R$.

For step2 we do some optimizations:
- Instead of applying matrix multiplications: $Q = I \times G_1^T \times G_2^T \times ... \times G_n^T$ and $R = G_n \times ... \times G_2 \times G_1 \times A$, we apply Gi vens rotations directly to the rows of $R$ and $Q$. We use eigen's vectorized operations instead of dual loops.
- Instead of computing the transpose of the rotation matrix, we apply the rotation to the columns of the matrix.

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

## MPI Optimization Results
We implemented MPI parallelization for two key components of our algorithm:
1. Random Matrix Generation in RSVD
2. Givens Rotation QR Decomposition

### Random Matrix Generation (2000x2000 matrix)
The following table shows the performance of parallel random matrix generation using different numbers of MPI processes:

| Number of Processes | Execution Time (seconds) | Speedup |
|--------------------|-------------------------|---------|
| 1                  | 0.891                  | 1.00x   |
| 2                  | 0.894                  | 1.00x   |
| 4                  | 0.761                  | 1.17x   |
| 8                  | 0.903                  | 0.99x   |

For random matrix generation, increasing the number of processes did not significantly improve performance. This is likely due to the communication overhead outweighing the computational benefits for this operation.

### Givens Rotation QR Decomposition (1000x1000 matrix)
The parallel implementation of Givens Rotation QR shows significant improvement with increased processes:

| Number of Processes | Execution Time (ms) | Speedup |
|--------------------|---------------------|---------|
| 1                  | 2128               | 1.00x   |
| 2                  | 1159               | 1.84x   |
| 4                  | 668                | 3.19x   |
| 8                  | 519                | 4.10x   |

The Givens Rotation QR decomposition shows excellent scaling with increased processes, achieving a 4.10x speedup with 8 processes. This demonstrates that our MPI implementation is particularly effective for computationally intensive matrix operations.

### Analysis
1. Random Matrix Generation:
   - The parallel implementation showed no significant speedup
   - Communication overhead likely dominates the computation time
   - Best performance achieved with single process

2. Givens Rotation QR:
   - Shows near-linear speedup up to 4 processes
   - Continues to improve with 8 processes
   - Demonstrates good parallel efficiency
   - Communication overhead is well-balanced with computational work

These results suggest that MPI parallelization is most effective for computationally intensive operations like QR decomposition, while simpler operations like random matrix generation may not benefit from parallelization due to communication overhead.

# RandomizedSVD algorithm overview
Randomized Singular Value Decomposition is a fast probabilistic algorithm that can be used to compute the near optimal low-rank singular value decomposition of massive data sets with high accuracy. The key idea is to compute a compressed representation of the data to capture the essential information. This compressed representation can then be used to obtain the low-rank singular value decomposition decomposition.
The larger the matrix, the higher the computational advantages of this algorithm are, considering that classical SVD computation requires 𝑂(𝑚𝑛 min(𝑚,𝑛)) operations. It's expecially efficient when the matrix is sparse or when only a small subset of singular values and vectors is needed.

The steps of the algorithm are:
1. Generate a projection S of the input matrix A on a random subspace, defined by a random Gaussian matrix, to reduce its column space dimensionality (rank) and capture its dominant structure;
2. Orthogonalize S by using QR decomposition;
3. Project A onto the subspace defined by Q to reduce the size of A. Call this projection B;
4. Compute a more efficient SVD on the smaller matrix B;
5. Recover the approximate SVD of A.

rSVD reaches a complexity of O(m*n*k)+O(k^2 *n)+O(k^3), where 
𝑘 is the reduced rank of A. This is faster than classical SVD if k<< min(m,n).

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