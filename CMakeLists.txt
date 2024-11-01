cmake_minimum_required(VERSION 3.22)
project(mpi_incompressible_fluid)


find_package(Python3 REQUIRED)
find_package(benchmark)


if (benchmark_FOUND)
	set(BENCH_OK TRUE)
endif()


set(CMAKE_CXX_STANDARD 23)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native -fopenmp-simd -fopt-info-all\
	-funroll-loops  -flto  -flto=auto -fno-signed-zeros -fno-trapping-math -ffinite-math-only -ffast-math -fpermissive \
	-faggressive-loop-optimizations -funsafe-math-optimizations -ftree-vectorizer-verbose=2 -fprofile-use")
#TODO: save profile in a know location so we can use it with multiple targets. -fprofile-generate to generate profile -fprofile-use to use it
#TODO: make the profile generation automatic when running the benchmark target before compiling the actual target, better if we can check if the profile is up to date or not and only regenerate it if needed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

if(NOT DEFINED VERBOSE)
    set(VERBOSE "FALSE")
    add_compile_definitions(VERBOSE=${VERBOSE})
endif ()


set(EXE_NAME "mif")
set(CONVERGENCE_TEST_EXE_NAME "convergence_test")
set(BENCHMARK_EXE_NAME "benchmark")

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src")
set(TEST_DIR    "${CMAKE_SOURCE_DIR}/test")

set(FORCING_TERM_GENERATOR ${CMAKE_SOURCE_DIR}/generator/manufsol.py)
set(FORCING_TERM_ARTIFACT ${CMAKE_SOURCE_DIR}/artifacts/manufsol.cpp)

set(CONVERGENCE_REPORT_NAME "convergence_report.csv")


include_directories(${INCLUDE_DIR})
add_library(
	miflib
	STATIC
	${SOURCE_DIR}/Constants.cpp
	${SOURCE_DIR}/Norms.cpp
	${SOURCE_DIR}/Timestep.cpp
	${SOURCE_DIR}/VectorFunction.cpp
	${SOURCE_DIR}/VelocityTensor.cpp
	${FORCING_TERM_ARTIFACT}
)


add_executable(${EXE_NAME} "${SOURCE_DIR}/main.cpp")
target_link_libraries(${EXE_NAME} PRIVATE miflib)

add_executable(${CONVERGENCE_TEST_EXE_NAME} "${TEST_DIR}/convergence.cpp")
target_link_libraries(${CONVERGENCE_TEST_EXE_NAME} PRIVATE miflib)

if (${BENCH_OK})
	add_executable(${BENCHMARK_EXE_NAME} "${TEST_DIR}/benchmark.cpp")
	target_link_libraries(${BENCHMARK_EXE_NAME} PRIVATE miflib benchmark::benchmark)

	add_custom_target(
		bench
		DEPENDS ${BENCHMARK_EXE_NAME}
		COMMAND ./${BENCHMARK_EXE_NAME} --benchmark_counters_tabular=true
	)
endif()

add_executable(tensor_usage_test test/tensor_usage_test.cpp)

add_custom_command(
	OUTPUT ${FORCING_TERM_ARTIFACT}
	DEPENDS ${FORCING_TERM_GENERATOR}
	COMMAND ${Python3_EXECUTABLE} ${FORCING_TERM_GENERATOR} > ${FORCING_TERM_ARTIFACT}
)

add_custom_command(
	OUTPUT ${CONVERGENCE_REPORT_NAME}
	DEPENDS ${CONVERGENCE_TEST_EXE_NAME}
	COMMAND ./${CONVERGENCE_TEST_EXE_NAME} > ${CONVERGENCE_REPORT_NAME}
)

add_custom_target(
	convergence
	DEPENDS ${CONVERGENCE_REPORT_NAME}
	COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/plot/convergence_plot.py ${CONVERGENCE_REPORT_NAME}
)
