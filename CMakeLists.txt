cmake_minimum_required(VERSION 3.27)
project(mpi_incompressible_fluid)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(NOT DEFINED VERBOSE)
    set(VERBOSE "FALSE")
    add_compile_definitions(VERBOSE=${VERBOSE})
endif ()
set(EXE_NAME "mif")

file(GLOB SOURCES "src/main.cpp")
include_directories(include)




add_executable(${EXE_NAME} ${SOURCES})
add_executable(tensor_usage src/tensor_usage.cpp)

