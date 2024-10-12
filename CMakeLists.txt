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
set(FORCING_TERM_ARTIFACT ${CMAKE_SOURCE_DIR}/artifacts/manufsol.cpp)

file(GLOB SOURCES "src/*.cpp")
include_directories(include)




add_executable(${EXE_NAME} ${SOURCES} ${FORCING_TERM_ARTIFACT})
add_executable(tensor_usage_test test/tensor_usage_test.cpp)

add_custom_command(
	OUTPUT ${FORCING_TERM_ARTIFACT}
	DEPENDS ${FORCING_TERM_GENERATOR}
	COMMAND ${Python3_EXECUTABLE} ${FORCING_TERM_GENERATOR} > ${FORCING_TERM_ARTIFACT}
)
