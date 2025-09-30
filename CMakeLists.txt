cmake_minimum_required(VERSION 3.14)
project(eigenwave VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create header-only library interface
add_library(eigenwave INTERFACE)
target_include_directories(eigenwave INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Enable testing if building tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples optionally
if(BUILD_EXAMPLES)
    add_executable(basic_usage examples/basic_usage.cpp)
    target_link_libraries(basic_usage PRIVATE eigenwave)

    add_executable(slicing_broadcasting_demo examples/slicing_broadcasting_demo.cpp)
    target_link_libraries(slicing_broadcasting_demo PRIVATE eigenwave)

    add_executable(concatenate_demo examples/concatenate_demo.cpp)
    target_link_libraries(concatenate_demo PRIVATE eigenwave)

    add_executable(matmul_demo examples/matmul_demo.cpp)
    target_link_libraries(matmul_demo PRIVATE eigenwave)
endif()