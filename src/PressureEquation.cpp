#include "PressureEquation.h"
#include "StaggeredTensorMacros.h"
#include "VelocityDivergence.h"
#include <fftw3.h>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop

namespace mif {
    
    inline Real compute_eigenvalue_neumann(size_t index, size_t N) {
        return (std::cos(M_PI * index / (N-1)) - 1.0) * (N-1) * (N-1) / (2.0*M_PI*M_PI);
    }

    void solve_pressure_equation_neumann(StaggeredTensor &pressure,
                                         const VelocityTensor &velocity) {
        const Constants &constants = velocity.constants;

        // Fill the rhs buffer with the divergence of the velocity.
        STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(pressure, true, pressure(i,j,k) = calculate_velocity_divergence(velocity,i,j,k) / constants.dt;)

        // Create 2decomp object.
        bool neumannBC[3] = {true, true, true};
        C2Decomp c2d = C2Decomp(constants.Nx, constants.Ny, constants.Nz, constants.Pz, constants.Py, neumannBC);

        // Execute type 1 DCT along direction x and transpose from (z,y,x) to (x,z,y).
        // Note: major refers to the last index of the triple, the triple is in the iterating order.
        // TODO: only allocate space once.
        Real *temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        Real *temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        fftw_plan fft_plan = fftw_plan_r2r_1d(constants.Nz, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t k = 0; k < constants.Nz; k++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                const size_t base_index = constants.Nx*constants.Ny * k + constants.Nx * j;

                // Copy the original data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    temp1[i] = pressure(base_index+i);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    pressure(base_index+i) = temp2[i];
                }
            }
        }
        c2d.transposeX2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));

        fftw_free(temp1);
        fftw_free(temp2);


        // Execute type 1 DCT along direction y and transpose from (x,z,y) to (y,x,z).
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        fft_plan = fftw_plan_r2r_1d(constants.Ny, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                const size_t base_index = constants.Nz*constants.Ny * i + constants.Ny * k;

                // Copy the original data.
                for (size_t j = 0; j < constants.Ny; j++) {
                    temp1[j] = pressure(base_index+j);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t j = 0; j < constants.Ny; j++){
                    pressure(base_index+j) = temp2[j];
                }
            }
        }
        c2d.transposeY2Z_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));

        fftw_free(temp1);
        fftw_free(temp2);


        // Execute type 1 DCT along direction z while in indexing (y,x,z), do not transpose.
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        fft_plan = fftw_plan_r2r_1d(constants.Nz, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                const size_t base_index = constants.Nz*constants.Nx * j + constants.Nz * i;

                // Copy the original data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    temp1[k] = pressure(base_index+k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    pressure(base_index+k) = temp2[k];
                }
            }
        }

        fftw_free(temp1);
        fftw_free(temp2);


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
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        fft_plan = fftw_plan_r2r_1d(constants.Nz, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                const size_t base_index = constants.Nz*constants.Nx * j + constants.Nz * i;

                // Copy the original data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    temp1[k] = pressure(base_index+k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    pressure(base_index+k) = temp2[k] / (2.0*(constants.Nz-1));
                }
            }
        }
        c2d.transposeZ2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));

        fftw_free(temp1);
        fftw_free(temp2);


        // Execute type 1 IDCT along direction y and transpose from (x,z,y) to (z,y,x).
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        fft_plan = fftw_plan_r2r_1d(constants.Ny, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                const size_t base_index = constants.Nz*constants.Ny * i + constants.Ny * k;

                // Copy the original data.
                for (size_t j = 0; j < constants.Ny; j++) {
                    temp1[j] = pressure(base_index+j);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t j = 0; j < constants.Ny; j++){
                    pressure(base_index+j) = temp2[j] / (2.0*(constants.Ny-1));
                }
            }
        }
        c2d.transposeY2X_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


        // Execute type 1 IDCT along direction x while in indexing (z,y,x), do not transpose.
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        fft_plan = fftw_plan_r2r_1d(constants.Nx, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t k = 0; k < constants.Nz; k++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                const size_t base_index = constants.Nx*constants.Ny * k + constants.Nx * j;

                // Copy the original data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    temp1[i] = pressure(base_index+i);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    pressure(base_index+i) = temp2[i] / (2.0*(constants.Nx-1));
                }
            }
        }

        fftw_free(temp1);
        fftw_free(temp2);


        fftw_destroy_plan(fft_plan);
        fftw_cleanup();
    }
}