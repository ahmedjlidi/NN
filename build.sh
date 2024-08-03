#!/bin/bash

rm NN

rm -rf build/*

# Ensure build directory exists
mkdir -p build

# Navigate to build directory
cd build

# Run CMake with Release build type
cmake -DCMAKE_BUILD_TYPE=Release ..

# Clean previous builds
make clean

# Build the project with parallel compilation
make -j$(nproc)

# Move the executable to the parent directory
mv NN ..