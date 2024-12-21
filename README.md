# Randomized SVD
- [Overview](#overview)
- [Programming Paradigm](#programming-paradigm)
- [Algorithms description](#algorithms-description)
  + [PowerMethod SVD](#powermethod-svd)
  + [Givens Rotation QR Decomposition](#givens-rotation-qr-decomposition)
  + [RandomizedSVD](#randomizedsvd)
- [Project setup](#project-setup)
  + [MacOS setup](#macos-setup)
  + [Windows WSL2 setup](#windows-wsl2-setup)
  + [Compiler setup](#compiler-setup)
  + [Build the project](#build-the-project)
- [Benchmarks](#benchmarks)
- [MPI & OMP Example](#mpi--omp-example)

## Overview
This project is a C++ implementation of the **Randomized Singular Value Decomposition (rSVD)** algorithm. We only used the matrix operations of the `Eigen` library to implement our algorithm. We do some benchmarks to compare the performance of our implementation with the `Eigen` library, the result indicates that our implementation can enhance the performance of handling large or sparse matrices.

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

### RandomizedSVD
Randomized Singular Value Decomposition is a fast probabilistic algorithm that can be used to compute the near optimal low-rank singular value decomposition of massive data sets with high accuracy. The key idea is to compute a compressed representation of the data to capture the essential information. This compressed representation can then be used to obtain the low-rank singular value decomposition decomposition.
The larger the matrix, the higher the computational advantages of this algorithm are, considering that classical SVD computation requires 𝑂(𝑚𝑛 min(𝑚,𝑛)) operations. It's expecially efficient when the matrix is sparse or when only a small subset of singular values and vectors is needed.

The steps of the algorithm are:
1. Draw an $n \times k$ Gaussian random matrix $\Omega$
2. From the $m \times k$ sketch matrix $Y=A \Omega$
3. From an $m \times k$ orthonormal matrix $Q$ such that $Y=QR$
4. From the $n \times k$ matrix $B=Q^TA$
5. Compute the SVD of $\hat{U} \Sigma V^{T}$
6. From the matrix $U=Q\hat{U}$

rSVD reaches a complexity of $O(m \times n \times k) + O(k^2 \times n) + O(k^3)$, where
$k$ is the reduced rank of $A$. This is faster than classical SVD if $k<< min(m,n)$.


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

The following figure summarizes the results of givens rotation QR for sparse matrices:
![alt text](figures/householder_givens.jpg)
This figure shows we should do more to optimize the givens rotation QR for sparse matrices.

The following figure summarizes the results of multiple SVD method for sparse matrices:
![alt text](figures/plot.jpg)

## MPI & OMP Example
```bash
mpirun -np 4 ./build/profiling/mpi_omp_random_matrix
```
We have implemented benchmarks to compare the performance to generate random matrix using MPI and OpenMP. The following table summarizes the results:

| Configuration     | Number of Processes | Time Taken (seconds) |
|-------------------|---------------------|----------------------|
| RandomMTX mpi     | 1                   | 0.954016             |
| RandomMTX mpi     | 2                   | 0.686535             |
| RandomMTX mpi     | 4                   | 0.594157             |
| RandomMTX mpi_omp | 1                   | 0.998558             |
| RandomMTX mpi_omp | 2                   | 3.01405              |
| RandomMTX mpi_omp | 4                   | 0.635416             |
| RandomMTX omp     | 1                   | ***0.0920939***      |

![alt text](figures/image.png)
### Analysis
- Notably, the "RandomMTX omp" configuration with 1 process shows an outstanding performance with a time taken of just 0.0920939 seconds, which is the fastest.
- Excessive parallelization may lead to a huge amount of creation and destruction of resources, and as a result, the performance may turn out to be worse.