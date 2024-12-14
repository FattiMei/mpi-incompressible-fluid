cmake_minimum_required(VERSION 3.22)
project(mpi_incompressible_fluid)


find_package(Python3 REQUIRED)
find_package(benchmark)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries(PkgConfig::FFTW)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

if (benchmark_FOUND)
	set(BENCH_OK TRUE)
endif()


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -mtune=native \
	-funroll-all-loops -flto -fno-signed-zeros -fno-trapping-math -flto=auto -DNDEBUG") #-DNDEBUG
# TODO: split into Debug and Release (with -DNDEBUG in Release)


if(NOT DEFINED VERBOSE)
    set(VERBOSE "FALSE")
    add_compile_definitions(VERBOSE=${VERBOSE})
endif ()

#add_definitions(-DOPT_CPU_CACHE)

set(EXE_NAME "mif")
set(CONVERGENCE_TEST_EXE_NAME "convergence_test")
set(BENCHMARK_EXE_NAME "benchmark")
set(MPI_TEST_EXE_NAME "mpi_test")
set(VELOCITY_TEST_EXE_NAME "velocity_test")

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src")
set(TEST_DIR    "${CMAKE_SOURCE_DIR}/test")

set(DECOMP_DIR "${CMAKE_SOURCE_DIR}/deps/2Decomp_C")

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


include_directories(${DECOMP_DIR})
add_library(
	decomp
	STATIC
	${DECOMP_DIR}/C2Decomp.cpp
	${DECOMP_DIR}/Alloc.cpp
	${DECOMP_DIR}/TransposeX2Y.cpp
	${DECOMP_DIR}/TransposeY2Z.cpp
	${DECOMP_DIR}/TransposeZ2Y.cpp
	${DECOMP_DIR}/TransposeY2X.cpp
	${DECOMP_DIR}/MemSplitMerge.cpp
	${DECOMP_DIR}/IO.cpp
	${DECOMP_DIR}/Best2DGrid.cpp
	${DECOMP_DIR}/Halo.cpp
)

add_executable(${EXE_NAME} "${SOURCE_DIR}/main.cpp")
target_link_libraries(${EXE_NAME} PRIVATE miflib ${MPI_C_LIBRARIES})

add_executable(${CONVERGENCE_TEST_EXE_NAME} "${TEST_DIR}/convergence.cpp")
target_link_libraries(${CONVERGENCE_TEST_EXE_NAME} PRIVATE miflib ${MPI_C_LIBRARIES})

add_executable(${MPI_TEST_EXE_NAME} "${TEST_DIR}/mpi_test.cpp")
target_link_libraries(${MPI_TEST_EXE_NAME} PRIVATE miflib ${MPI_C_LIBRARIES})

add_executable(${VELOCITY_TEST_EXE_NAME} "${TEST_DIR}/velocity_test.cpp")
target_link_libraries(${VELOCITY_TEST_EXE_NAME} PRIVATE miflib ${MPI_C_LIBRARIES})

if (${BENCH_OK})
	add_executable(${BENCHMARK_EXE_NAME} "${TEST_DIR}/benchmark.cpp")
	target_link_libraries(${BENCHMARK_EXE_NAME} PRIVATE miflib benchmark::benchmark ${MPI_C_LIBRARIES})

	add_custom_target(
		bench
		DEPENDS ${BENCHMARK_EXE_NAME}
		COMMAND ./${BENCHMARK_EXE_NAME} --benchmark_counters_tabular=true
	)
endif()

add_executable(tensor_usage_test test/tensor_usage_test.cpp)
## add_executable(fftw_Dirichlet test/fftw3D_Dirichlet.cpp)
add_executable(fftw_Dirichlet test/fftw.cpp)
add_executable(fftw_Neumann test/fftw3D_Neumann.cpp)

target_include_directories(decomp PUBLIC ${DECOMP_DIR})
target_link_libraries(fftw_Neumann PRIVATE decomp)
target_link_libraries(fftw_Dirichlet PRIVATE decomp)

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
