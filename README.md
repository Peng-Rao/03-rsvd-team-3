# Randomized SVD
## Project setup
We use `CMake` to build the project, and `Vcpkg` to manage dependencies, our project can run across platforms.

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

And then write building script, add to project root directory:
## CMakeUserPresets.json
```json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "default",
      "inherits": "vcpkg",
      "environment": {
        "VCPKG_ROOT": "xxxxxxxxxxxxxxxxxxxx/vcpkg"
      },
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      }
    }
  ]
}

```

Finally, configure the build using CMake:

1.  Configure the build using CMake:
```
cmake --preset=default
```
2. Build the project
```
cmake --build build
```
3. Run the application
```
./build/main
```
### or
```
mpirun -np 4 ./build/main
```

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