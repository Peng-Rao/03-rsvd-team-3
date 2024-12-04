# Project setup
For `macos`, install `CMake` and `Vcpkg`:
```bash
brew install cmake
brew install vcpkg
```
Add to `PATH`:
```bash
vim ~/.zshrc

# Add
export VCPKG_ROOT=/Users/{{your username}}/vcpkg
export PATH=$VCPKG_ROOT:$PATH
```

For Windows, see this guide: https://learn.microsoft.com/en-us/vcpkg/get_started/get-started/

**Fork** and **Clone** this project to your own repo.

Install libraries:
```
vcpkg install
```

And then write building script:
CMakeUserPresets.json
```json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "default",
      "inherits": "vcpkg",
      "environment": {
        "VCPKG_ROOT": "/Users/{{your username}}/vcpkg"
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