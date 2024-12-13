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
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export PATH="/usr/local/opt/llvm/bin:$PATH"
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