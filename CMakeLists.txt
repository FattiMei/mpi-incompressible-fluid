cmake_minimum_required(VERSION 3.20)
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

# Set the compiler.
set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_COMPILER "mpicxx")

# Choose build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 23)

# Set custom flags for Debug build.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -mtune=native \
    -funroll-all-loops -flto -fno-signed-zeros -fno-trapping-math -flto=auto")

# Set custom flags for Release build.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -DNDEBUG")


if(NOT DEFINED VERBOSE)
    set(VERBOSE "FALSE")
    add_compile_definitions(VERBOSE=${VERBOSE})
endif ()

#add_definitions(-DOPT_CPU_CACHE)
add_definitions(-DMIF_LEGACY_COMPILER)

get_directory_property(MYDEFS COMPILE_DEFINITIONS)
if (MYDEFS MATCHES "MIF_LEGACY_COMPILER")
	MESSAGE(WARNING "Using legacy endian conversion")
endif()

set(EXE_NAME "mif")
set(CONVERGENCE_TEST_EXE_NAME "convergence_test")
set(BENCHMARK_EXE_NAME "benchmark")

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src")
set(TEST_DIR    "${CMAKE_SOURCE_DIR}/test")

set(DECOMP_DIR "${CMAKE_SOURCE_DIR}/deps/2Decomp_C")

set(FORCING_TERM_GENERATOR ${CMAKE_SOURCE_DIR}/generators/manufsol_velocity.py)
set(FORCING_TERM_ARTIFACT ${CMAKE_SOURCE_DIR}/artifacts/manufsol_velocity.cpp)
set(PRESSURE_GENERATOR ${CMAKE_SOURCE_DIR}/generators/manufsol_pressure.py)
set(PRESSURE_ARTIFACT ${CMAKE_SOURCE_DIR}/artifacts/manufsol_pressure.cpp)
set(EXACT_SOLUTION_GENERATOR ${CMAKE_SOURCE_DIR}/generators/manufsol.py)
set(EXACT_SOLUTION_ARTIFACT ${CMAKE_SOURCE_DIR}/artifacts/manufsol.cpp)

set(CONVERGENCE_REPORT_NAME "convergence_report.csv")

include_directories(${DECOMP_DIR})
add_library(
	decomp
	STATIC
	${DECOMP_DIR}/Alloc.cpp
	${DECOMP_DIR}/Best2DGrid.cpp
	${DECOMP_DIR}/C2Decomp.cpp
	${DECOMP_DIR}/Halo.cpp
	${DECOMP_DIR}/IO.cpp
	${DECOMP_DIR}/MemSplitMerge.cpp
	${DECOMP_DIR}/TransposeX2Y.cpp
	${DECOMP_DIR}/TransposeY2X.cpp
	${DECOMP_DIR}/TransposeY2Z.cpp
	${DECOMP_DIR}/TransposeZ2Y.cpp
)
target_compile_options(decomp PUBLIC -w) # Ignore all warnings from 2decomp.

include_directories(${INCLUDE_DIR})
add_library(
	miflib
	STATIC
	${SOURCE_DIR}/Constants.cpp
	${SOURCE_DIR}/Norms.cpp
	${SOURCE_DIR}/PressureEquation.cpp
	${SOURCE_DIR}/PressureSolverStructures.cpp
	${SOURCE_DIR}/PressureTensor.cpp
	${SOURCE_DIR}/StaggeredTensor.cpp
	${SOURCE_DIR}/Timestep.cpp
	${SOURCE_DIR}/TimestepVelocity.cpp
	${SOURCE_DIR}/VectorFunction.cpp
	${SOURCE_DIR}/VelocityTensor.cpp
	${SOURCE_DIR}/InputParser.cpp
	${SOURCE_DIR}/VTKDatExport.cpp
	${FORCING_TERM_ARTIFACT}
	${PRESSURE_ARTIFACT}
	${EXACT_SOLUTION_ARTIFACT}
)

# Add all executables.
add_executable(${EXE_NAME} "${SOURCE_DIR}/main.cpp")
target_link_libraries(${EXE_NAME} PRIVATE miflib decomp ${MPI_C_LIBRARIES})

add_executable(${CONVERGENCE_TEST_EXE_NAME} "${TEST_DIR}/convergence.cpp")
target_link_libraries(${CONVERGENCE_TEST_EXE_NAME} PRIVATE miflib ${MPI_C_LIBRARIES})

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

add_executable(velocity_test test/velocity_test.cpp)
add_executable(velocity_test_mixed test/velocity_test_mixed.cpp)

add_executable(mpi_test test/mpi_test.cpp)

add_executable(fftw_Dirichlet3D test/fftw3D_Dirichlet.cpp)
add_executable(fftw_Dirichlet1D test/fftw1D_Dirichlet.cpp)
add_executable(fftw_Dirichlet1D_spectral test/fftw1D_Dirichlet_spectral.cpp)
add_executable(fftw_Neumann3D test/fftw3D_Neumann.cpp)
add_executable(fftw_Neumann1D test/fftw1D_Neumann.cpp)
add_executable(2decomp_test test/2decomp_test.cpp)
add_executable(2decomp_test_parallel test/2decomp_test_parallel.cpp)

# temporary: here are the tests that have been properly validated
add_executable(test_fftw_1d_periodic test/fftw/test_fftw_1d_periodic.cpp)
add_executable(test_fftw_1d_neumann test/fftw/test_fftw_1d_neumann.cpp)
add_executable(test_fftw_1d_neumann_in_place test/fftw/test_fftw_1d_neumann_in_place.cpp)

add_executable(test_fftw_3d_neumann test/fftw/test_fftw_3d_neumann.cpp)
target_link_libraries(test_fftw_3d_neumann PRIVATE decomp)

add_executable(test_2decomp_reverse_engineer test/2decomp/test_2decomp_reverse_engineer.cpp)
target_link_libraries(test_2decomp_reverse_engineer PRIVATE decomp)

add_executable(pressure_test_hn test/pressure_test_hn.cpp)
add_executable(pressure_test_nhn test/pressure_test_nhn.cpp)
add_executable(pressure_test_mixed test/pressure_test_mixed.cpp)

add_executable(full_test test/full_test.cpp)


# Set link libraries.
target_link_libraries(tensor_usage_test PRIVATE miflib)

target_link_libraries(velocity_test PRIVATE miflib)
target_link_libraries(velocity_test_mixed PRIVATE miflib)

target_link_libraries(fftw_Dirichlet3D PRIVATE decomp)
target_link_libraries(fftw_Dirichlet1D PRIVATE decomp)
target_link_libraries(fftw_Dirichlet1D_spectral PRIVATE decomp)
target_link_libraries(fftw_Neumann3D PRIVATE decomp)
target_link_libraries(fftw_Neumann1D PRIVATE decomp)
target_link_libraries(2decomp_test PRIVATE miflib decomp)
target_link_libraries(2decomp_test_parallel PRIVATE miflib decomp)

target_link_libraries(pressure_test_hn PRIVATE miflib decomp)
target_link_libraries(pressure_test_nhn PRIVATE miflib decomp)
target_link_libraries(pressure_test_mixed PRIVATE miflib decomp)

target_link_libraries(full_test PRIVATE miflib decomp)

add_custom_command(
	OUTPUT ${FORCING_TERM_ARTIFACT}
	DEPENDS ${FORCING_TERM_GENERATOR}
	COMMAND ${Python3_EXECUTABLE} ${FORCING_TERM_GENERATOR} > ${FORCING_TERM_ARTIFACT}
)
add_custom_command(
	OUTPUT ${EXACT_SOLUTION_ARTIFACT}
	DEPENDS ${EXACT_SOLUTION_GENERATOR}
	COMMAND ${Python3_EXECUTABLE} ${EXACT_SOLUTION_GENERATOR} > ${EXACT_SOLUTION_ARTIFACT}
)
add_custom_command(
	OUTPUT ${PRESSURE_ARTIFACT}
	DEPENDS ${PRESSURE_GENERATOR}
	COMMAND ${Python3_EXECUTABLE} ${PRESSURE_GENERATOR} > ${PRESSURE_ARTIFACT}
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
