# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.20)

# Project name
project(mpi_incompressible_fluid_tests)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)

# Include the directories for the main project and Google Test
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/test)

# Find the Google Test package
find_package(GTest REQUIRED)

# Add the test source files
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp")

# Create an executable for the tests
add_executable(runTests ${TEST_SOURCES})

# Link the Google Test libraries and the main project libraries
target_link_libraries(runTests GTest::GTest GTest::Main)