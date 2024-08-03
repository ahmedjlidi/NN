cmake_minimum_required(VERSION 3.22.1)
project(NN)

# Specify the C and C++ compilers
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Add source files to the project
set(SOURCE_FILES
        main.cpp
        Utility.cpp
        stdafx.cpp
        Timer.cpp
        Tensor.cpp
        Operators.cpp
        Dataset.cpp
        Ann.cpp
        Ann2.cpp
        Networks/Resources/Layer.cpp
        Networks/Resources/Net.cpp
        Networks/BNet.cpp
)

# Add executable
add_executable(NN ${SOURCE_FILES})

# Add the precompiled header setup
target_precompile_headers(NN PRIVATE stdafx.h)
