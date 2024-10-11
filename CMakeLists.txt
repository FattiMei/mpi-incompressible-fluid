cmake_minimum_required(VERSION 3.22)
project(mpi_incompressible_fluid)

find_package(Python3 REQUIRED)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(NOT DEFINED VERBOSE)
    set(VERBOSE "FALSE")
    add_compile_definitions(VERBOSE=${VERBOSE})
endif ()
set(EXE_NAME "mif")
set(FORCING_TERM_GENERATOR ${CMAKE_SOURCE_DIR}/generator/manufsol.py)

file(GLOB SOURCES "src/main.cpp")
include_directories(include)




add_executable(${EXE_NAME} ${SOURCES})
add_executable(tensor_usage src/tensor_usage.cpp)

add_custom_target(
	manifacture ALL
	DEPENDS ${FORCING_TERM_GENERATOR}
	COMMAND ${Python3_EXECUTABLE} ${FORCING_TERM_GENERATOR} > ${CMAKE_SOURCE_DIR}/artifacts/manufsol.cpp
)
