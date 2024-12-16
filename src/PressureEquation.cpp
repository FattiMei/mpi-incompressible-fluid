#include "PressureEquation.h"
#include "PressureSolverStructures.h"
#include "StaggeredTensorMacros.h"
#include "VelocityDivergence.h"

namespace mif {
    
    inline Real compute_eigenvalue_neumann(size_t index, size_t N) {
        return (std::cos(M_PI * index / (N-1)) - 1.0) * (N-1) * (N-1) / (2.0*M_PI*M_PI);
    }

    void solve_pressure_equation_neumann(StaggeredTensor &pressure,
                                         const VelocityTensor &velocity,
                                         PressureSolverStructures &structures) {
        const Constants &constants = velocity.constants;

        // Fill the rhs buffer with the divergence of the velocity.
        STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(pressure, true, pressure(i,j,k) = calculate_velocity_divergence(velocity,i,j,k) / constants.dt;)

        // Execute type 1 DCT along direction x and transpose from (z,y,x) to (x,z,y).
        // Note: major refers to the last index of the triple, the triple is in the iterating order.
        for (size_t k = 0; k < constants.Nz; k++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                const size_t base_index = constants.Nx*constants.Ny * k + constants.Nx * j;

                // Copy the original data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    structures.buffer_x[i] = pressure(base_index+i);
                }

                // Execute the fft.
                fftw_execute(structures.fft_plan_x);

                // Copy the transformed data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    pressure(base_index+i) = structures.buffer_x[i];
                }
            }
        }
        structures.c2d.transposeX2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


        // Execute type 1 DCT along direction y and transpose from (x,z,y) to (y,x,z).
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                const size_t base_index = constants.Nz*constants.Ny * i + constants.Ny * k;

                // Copy the original data.
                for (size_t j = 0; j < constants.Ny; j++) {
                    structures.buffer_y[j] = pressure(base_index+j);
                }

                // Execute the fft.
                fftw_execute(structures.fft_plan_y);

                // Copy the transformed data.
                for (size_t j = 0; j < constants.Ny; j++){
                    pressure(base_index+j) = structures.buffer_y[j];
                }
            }
        }
        structures.c2d.transposeY2Z_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


        // Execute type 1 DCT along direction z while in indexing (y,x,z), do not transpose.
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                const size_t base_index = constants.Nz*constants.Nx * j + constants.Nz * i;

                // Copy the original data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    structures.buffer_z[k] = pressure(base_index+k);
                }

                // Execute the fft.
                fftw_execute(structures.fft_plan_z);

                // Copy the transformed data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    pressure(base_index+k) = structures.buffer_z[k];
                }
            }
        }


        // Divide by eigenvalues.
        for (size_t j = 0; j < constants.Ny; j++) {
            const Real lambda_2 = compute_eigenvalue_neumann(j, constants.Ny);
            for (size_t i = 0; i < constants.Nx; i++) {
                const Real lambda_1= compute_eigenvalue_neumann(i, constants.Nx);
                for (size_t k = 0; k < constants.Nz; k++) {
                    const Real lambda_3 = compute_eigenvalue_neumann(k, constants.Nz);
                    pressure(i,j,k) /= (lambda_1 + lambda_2 + lambda_3);
                }
            }
        }
        pressure(0,0,0) = 0;


        // Execute type 1 IDCT along direction z, transpose from (y,x,z) to (x,z,y).
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                const size_t base_index = constants.Nz*constants.Nx * j + constants.Nz * i;

                // Copy the original data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    structures.buffer_z[k] = pressure(base_index+k);
                }

                // Execute the fft.
                fftw_execute(structures.fft_plan_z);

                // Copy the transformed data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    pressure(base_index+k) = structures.buffer_z[k] / (2.0*(constants.Nz-1));
                }
            }
        }
        structures.c2d.transposeZ2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


        // Execute type 1 IDCT along direction y and transpose from (x,z,y) to (z,y,x).
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                const size_t base_index = constants.Nz*constants.Ny * i + constants.Ny * k;

                // Copy the original data.
                for (size_t j = 0; j < constants.Ny; j++) {
                    structures.buffer_y[j] = pressure(base_index+j);
                }

                // Execute the fft.
                fftw_execute(structures.fft_plan_y);

                // Copy the transformed data.
                for (size_t j = 0; j < constants.Ny; j++){
                    pressure(base_index+j) = structures.buffer_y[j] / (2.0*(constants.Ny-1));
                }
            }
        }
        structures.c2d.transposeY2X_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


        // Execute type 1 IDCT along direction x while in indexing (z,y,x), do not transpose.
        for (size_t k = 0; k < constants.Nz; k++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                const size_t base_index = constants.Nx*constants.Ny * k + constants.Nx * j;

                // Copy the original data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    structures.buffer_x[i] = pressure(base_index+i);
                }

                // Execute the fft.
                fftw_execute(structures.fft_plan_x);

                // Copy the transformed data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    pressure(base_index+i) = structures.buffer_x[i] / (2.0*(constants.Nx-1));
                }
            }
        }
    }
}