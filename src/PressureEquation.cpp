#include "PressureEquation.h"
#include <cassert>
#include "PressureSolverStructures.h"
#include "StaggeredTensorMacros.h"
#include "VelocityDivergence.h"

namespace mif {

// Add the non-homogeneous term to the boundaries of the rhs.
void apply_non_homogeneous_neumann(StaggeredTensor &rhs, const VectorFunction &exact_pressure_gradient) {
    const Constants &constants = rhs.constants;
    assert(!constants.periodic_bc[0] && !constants.periodic_bc[1] && !constants.periodic_bc[2]);
    std::array<size_t, 3> sizes = rhs.sizes();

    // Face 1: z=z_min
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, j, 0) += rhs.evaluate_function_at_index(i, j, 0, exact_pressure_gradient.f_w) * 2.0 * constants.one_over_dz;
        }
    }

    // Face 2: z=z_max
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, j, sizes[2]-1) -= rhs.evaluate_function_at_index(i, j, sizes[2]-1, exact_pressure_gradient.f_w) * 2.0 * constants.one_over_dz;
        }
    }

    // Face 3: y=y_min
    for (size_t k = 0; k < sizes[2]; k++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, 0, k) += rhs.evaluate_function_at_index(i, 0, k, exact_pressure_gradient.f_v) * 2.0 * constants.one_over_dy;
        }
    }

    // Face 4: y=y_max
    for (size_t k = 0; k < sizes[2]; k++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, sizes[1]-1, k) -= rhs.evaluate_function_at_index(i, sizes[1]-1, k, exact_pressure_gradient.f_v) * 2.0 * constants.one_over_dy;
        }
    }

    // Face 5: x=x_min
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
            rhs(0, j, k) += rhs.evaluate_function_at_index(0, j, k, exact_pressure_gradient.f_u) * 2.0 * constants.one_over_dx;
        }
    }

    // Face 6: x=x_max
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
            rhs(sizes[0]-1, j, k) -= rhs.evaluate_function_at_index(sizes[0]-1, j, k, exact_pressure_gradient.f_u) * 2.0 * constants.one_over_dx;
        }
    }
}

// Compute the rhs for homogeneous Neumann or periodic boundary conditions.
void compute_rhs_homogeneous_periodic(StaggeredTensor &rhs, const VelocityTensor &velocity, Real dt) {
    STAGGERED_TENSOR_ITERATE_OVER_ALL_OWNER_POINTS(rhs, rhs(i,j,k) = calculate_velocity_divergence(velocity,i,j,k) / dt;)
}
    
// Solve the pressure equation with Neumann or periodic boundary conditions.
// "pressure" contains the rhs, and will be replaced with the pressure.
void solve_pressure_equation(PressureTensor &pressure,    
                             const VelocityTensor &velocity) {
    const Constants &constants = velocity.constants;
    PressureSolverStructures &structures = pressure.structures;
    C2Decomp &c2d = structures.c2d;
    
#ifdef FFTW_USE_NEW_ARRAY_EXECUTE
      assert(fftw_alignment_of(static_cast<Real*>(pressure.raw_data())) == fftw_alignment_of(structures.buffer_x));
      assert(fftw_alignment_of(static_cast<Real*>(pressure.raw_data())) == fftw_alignment_of(structures.buffer_y));
      assert(fftw_alignment_of(static_cast<Real*>(pressure.raw_data())) == fftw_alignment_of(structures.buffer_z));
#endif

    // Execute type 1 DCT/FFT along direction x and transpose from (z,y,x) to (x,z,y).
    // Note: major refers to the last index of the triple, the triple is in the iterating order.
    for (int k = 0; k < c2d.xSize[2]; k++) {
        for (int j = 0; j < c2d.xSize[1]; j++) {
            const int base_index = c2d.xSize[0]*c2d.xSize[1] * k + c2d.xSize[0] * j;

#if defined(FFTW_USE_NEW_ARRAY_EXECUTE) && (USE_DOUBLE == 1)
            Real* ptr = static_cast<Real*>(pressure.raw_data()) + base_index;
            fftw_execute_r2r(structures.fft_plan_x, ptr, ptr);
#else
            // Copy the original data.
            memcpy(structures.buffer_x, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*c2d.xSize[0]);

            // Execute the fft.
#if USE_DOUBLE
            fftw_execute(structures.fft_plan_x);
#else
            fftwf_execute(structures.fft_plan_x);
#endif

            // Copy the transformed data.
            memcpy(static_cast<Real *>(pressure.raw_data())+base_index, structures.buffer_x, sizeof(Real)*c2d.xSize[0]);
#endif
        }
    }
    structures.c2d.transposeX2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 DCT/FFT along direction y and transpose from (x,z,y) to (y,x,z).
    for (int i = 0; i < c2d.ySize[0]; i++) {
        for (int k = 0; k < c2d.ySize[2]; k++) {
            const int base_index = c2d.ySize[2]*c2d.ySize[1] * i + c2d.ySize[1] * k;

#if defined(FFTW_USE_NEW_ARRAY_EXECUTE) && (USE_DOUBLE == 1)
            Real* ptr = static_cast<Real*>(pressure.raw_data()) + base_index;
            fftw_execute_r2r(structures.fft_plan_y, ptr, ptr);
#else
            // Copy the original data.
            memcpy(structures.buffer_y, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*c2d.ySize[1]);

            // Execute the fft.
#if USE_DOUBLE
            fftw_execute(structures.fft_plan_y);
#else
            fftwf_execute(structures.fft_plan_y);
#endif

            // Copy the transformed data.
            memcpy(static_cast<Real *>(pressure.raw_data())+base_index, structures.buffer_y, sizeof(Real)*c2d.ySize[1]);
#endif
        }
    }
    structures.c2d.transposeY2Z_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 DCT/FFT along direction z while in indexing (y,x,z), do not transpose.
    for (int j = 0; j < c2d.zSize[1]; j++) {
        for (int i = 0; i < c2d.zSize[0]; i++) {
            const int base_index = c2d.zSize[2]*c2d.zSize[0] * j + c2d.zSize[2] * i;

#if defined(FFTW_USE_NEW_ARRAY_EXECUTE) && (USE_DOUBLE == 1)
            Real* ptr = static_cast<Real*>(pressure.raw_data()) + base_index;
            fftw_execute_r2r(structures.fft_plan_z, ptr, ptr);
#else
            // Copy the original data.
            memcpy(structures.buffer_z, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*c2d.zSize[2]);

            // Execute the fft.
#if USE_DOUBLE
            fftw_execute(structures.fft_plan_z);
#else
            fftwf_execute(structures.fft_plan_z);
#endif
            // Copy the transformed data.
            memcpy(static_cast<Real *>(pressure.raw_data())+base_index, structures.buffer_z, sizeof(Real)*c2d.zSize[2]);
#endif
        }
    }


    // Divide by eigenvalues.
    for (int i = 0; i < c2d.zSize[0]*c2d.zSize[1]*c2d.zSize[2]; i++) {
        pressure(i) *= structures.eigenvalues(i);
    }
    if (constants.y_rank == 0 && constants.z_rank == 0) {
        pressure(0) = 0;
    }


    // Execute type 1 IDCT/IFFT along direction z, transpose from (y,x,z) to (x,z,y).
    const Real normalization_constant_z = constants.Nz_domains_global * (structures.periodic_bc[2] ? 1.0 : 2.0);
    for (int j = 0; j < c2d.zSize[1]; j++) {
        for (int i = 0; i < c2d.zSize[0]; i++) {
            const int base_index = c2d.zSize[2]*c2d.zSize[0] * j + c2d.zSize[2] * i;

#if defined(FFTW_USE_NEW_ARRAY_EXECUTE) && (USE_DOUBLE == 1)
            Real* ptr = static_cast<Real*>(pressure.raw_data()) + base_index;
            fftw_execute_r2r(structures.ifft_plan_z, ptr, ptr);
            // Normalize the transformed data.
            for (int k = 0; k < c2d.zSize[2]; k++) {
                pressure(base_index+k) /= normalization_constant_z;
            }
#else
            // Copy the original data.
            memcpy(structures.buffer_z, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*c2d.zSize[2]);

            // Execute the fft.
#if USE_DOUBLE
            fftw_execute(structures.ifft_plan_z);
#else
            fftwf_execute(structures.ifft_plan_z);
#endif
            // Copy the transformed data.
            for (int k = 0; k < c2d.zSize[2]; k++) {
                pressure(base_index+k) = structures.buffer_z[k] / normalization_constant_z;
            }
#endif
        }
    }
    structures.c2d.transposeZ2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 IDCT/IFFT along direction y and transpose from (x,z,y) to (z,y,x).
    const Real normalization_constant_y = constants.Ny_domains_global * (structures.periodic_bc[1] ? 1.0 : 2.0);
    for (int i = 0; i < c2d.ySize[0]; i++) {
        for (int k = 0; k < c2d.ySize[2]; k++) {
            const int base_index = c2d.ySize[2]*c2d.ySize[1] * i + c2d.ySize[1] * k;

#if defined(FFTW_USE_NEW_ARRAY_EXECUTE) && (USE_DOUBLE == 1)
            Real* ptr = static_cast<Real*>(pressure.raw_data()) + base_index;
            fftw_execute_r2r(structures.ifft_plan_y, ptr, ptr);
            // Normalize the transformed data.
            for (int j = 0; j < c2d.ySize[1]; j++){
                pressure(base_index+j) /= normalization_constant_y;
            }
#else
            // Copy the original data.
            memcpy(structures.buffer_y, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*c2d.ySize[1]);

            // Execute the fft.
#if USE_DOUBLE
            fftw_execute(structures.ifft_plan_y);
#else
            fftwf_execute(structures.ifft_plan_y);
#endif

            // Copy the transformed data.
            for (int j = 0; j < c2d.ySize[1]; j++){
                pressure(base_index+j) = structures.buffer_y[j] / normalization_constant_y;
            }
#endif
        }
    }
    structures.c2d.transposeY2X_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 IDCT/IFFT along direction x while in indexing (z,y,x), do not transpose.
    const Real normalization_constant_x = constants.Nx_domains * (structures.periodic_bc[0] ? 1.0 : 2.0);
    for (int k = 0; k < c2d.xSize[2]; k++) {
        for (int j = 0; j < c2d.xSize[1]; j++) {
            const int base_index = c2d.xSize[0]*c2d.xSize[1] * k + c2d.xSize[0] * j;

#if defined(FFTW_USE_NEW_ARRAY_EXECUTE) && (USE_DOUBLE == 1)
            Real* ptr = static_cast<Real*>(pressure.raw_data()) + base_index;
            fftw_execute_r2r(structures.ifft_plan_x, ptr, ptr);
            // Normalize the transformed data.
            for (int i = 0; i < c2d.xSize[0]; i++) {
                pressure(base_index+i) /= normalization_constant_x;
            }
#else
            // Copy the original data.
            memcpy(structures.buffer_x, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*c2d.xSize[0]);

            // Execute the fft.
#if USE_DOUBLE
            fftw_execute(structures.ifft_plan_x);
#else
            fftwf_execute(structures.ifft_plan_x);
#endif

            // Copy the transformed data.
            for (int i = 0; i < c2d.xSize[0]; i++) {
                pressure(base_index+i) = structures.buffer_x[i] / normalization_constant_x;
            }
#endif
        }
    }
}

void solve_pressure_equation_homogeneous_periodic(StaggeredTensor &pressure, 
                                                  PressureTensor &pressure_buffer,
                                                  const VelocityTensor &velocity,
                                                  Real dt) {
    compute_rhs_homogeneous_periodic(pressure, velocity, dt);
    pressure_buffer.copy_from_staggered(pressure);
    solve_pressure_equation(pressure_buffer, velocity);
    pressure_buffer.copy_to_staggered(pressure, 100);
}

void solve_pressure_equation_non_homogeneous_neumann(StaggeredTensor &pressure, 
                                                     PressureTensor &pressure_buffer,
                                                     const VelocityTensor &velocity,
                                                     const VectorFunction &exact_pressure_gradient,
                                                     Real dt) {
    compute_rhs_homogeneous_periodic(pressure, velocity, dt);
    apply_non_homogeneous_neumann(pressure, exact_pressure_gradient);
    pressure_buffer.copy_from_staggered(pressure);
    solve_pressure_equation(pressure_buffer, velocity);
    pressure_buffer.copy_to_staggered(pressure, 100);
}

void adjust_pressure(StaggeredTensor &pressure,
                     const std::function<Real(Real, Real, Real)> &exact_pressure) {
    const Constants &constants = pressure.constants;

    // Compute the sum of differences on each processor.
    Real local_difference = 0;
    STAGGERED_TENSOR_ITERATE_OVER_ALL_OWNER_POINTS(pressure, local_difference += 
            pressure.evaluate_function_at_index(i, j, k, exact_pressure) - pressure(i, j, k);)
    
    // Send the differences to the first processor. The first processor accumulates the differences and
    // sends back the result.
    Real difference = local_difference;
    if (constants.rank == 0) {
        // Receive the differences.
        for (int rank = 1; rank < constants.P; ++rank) {
            Real new_difference;
            MPI_Status status;
            int outcome = MPI_Recv(&new_difference, 1, MPI_MIF_REAL, rank, 0, MPI_COMM_WORLD, &status);
            assert(outcome == MPI_SUCCESS);
            (void) outcome;
            difference += new_difference;
        }

        // Compute the average difference.
        const size_t Nx = constants.periodic_bc[0] ? constants.Nx_global-1 : constants.Nx_global;
        const size_t Ny = constants.periodic_bc[1] ? constants.Ny_global-1 : constants.Ny_global;
        const size_t Nz = constants.periodic_bc[2] ? constants.Nz_global-1 : constants.Nz_global;
        difference /= (Nx * Ny * Nz);

        // Send the average difference to each processor.
        for (int rank = 1; rank < constants.P; ++rank) {
            int outcome = MPI_Send(&difference, 1, MPI_MIF_REAL, rank, 0, MPI_COMM_WORLD);
            assert(outcome == MPI_SUCCESS);
            (void) outcome;
        }
    } else {
        // Send the local difference to processor 0.
        int outcome = MPI_Send(&local_difference, 1, MPI_MIF_REAL, 0, 0, MPI_COMM_WORLD);
        assert(outcome == MPI_SUCCESS);

        // Receive the average difference from processor 0.
        MPI_Status status;
        outcome = MPI_Recv(&difference, 1, MPI_MIF_REAL, 0, 0, MPI_COMM_WORLD, &status);
        assert(outcome == MPI_SUCCESS);
        (void) outcome;
    }

    // Remove the difference from all points.
    for (size_t k = 0; k < constants.Nz; k++) {
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                pressure(i,j,k) += difference;
            }
        }
    }
}

} // mif
