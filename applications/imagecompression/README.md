# Image Compression using Randomized SVD

## Overview
This program compresses grayscale or color images using **Randomized Singular Value Decomposition (rSVD)**. It reduces the image dimensions while preserving the most important visual features. The compressed image is reconstructed using the top singular values, allowing for efficient storage with minimal perceptual loss.

## Dependencies
This program relies on:
- **Eigen** (for matrix operations)
- **stb_image** (for reading images)
- **stb_image_write** (for saving compressed images)

## Compilation Instructions
Ensure you have a C++ compiler installed (such as g++ or clang++). To compile the program, use:

```sh
 g++ -o image_compression ImageCompression.cpp -I<path_to_eigen> -I<path_to_stb> -std=c++17
```

Replace `<path_to_eigen>` and `<path_to_stb>` with the correct paths to the **Eigen** and **stb_image** header files.

## Running Instructions
Run the compiled program with:

```sh
./image_compression <image_path> <rank>
```

- `<image_path>`: Path to the input image.
- `<rank>`: The number of singular values to use for reconstruction.

## Expected Behavior
1. The program loads the image and detects if it is grayscale or RGB.
2. If the image is **visually grayscale** and has 3 channels, it converts it to a single-channel representation.
3. It applies **Randomized SVD (rSVD)** separately to each channel using the rank chosen.
4. The compressed image is reconstructed and saved as `compressed_image.png` in the output directory.

## Compression Characteristics
- Using a **rank below ~10%** of the image width significantly reduces memory usage, above this value, the compressed matrices occupy more memory than the original memory. Either way, it's suggested to combine this Image Compression function with a further compressing image saving format. Use formats thought for sparse and low-rank matrices.
- A **rank below ~5%** results in a visibly compressed image and might introduce artifacts.
- A **rank above ~15%** makes compression almost unnoticeable, retaining more details, but reduces the compression memory saving effect.

## Output
- The compressed image is saved as:
  ```
  ../test_image/compressed_image.png
  ```
  Modify this path in the source code if needed.


## Example Usage
For an image `imageRGB.png` and rank **50**:
```sh
./image_compression imageRGB.png 50
```
This would compress `imageRGB.png` using a rank of 50 for the rSVD algorithm and save the result in the `test_image` folder.

---

