#include "PressureEquation.h"
#include "VelocityDivergence.h"
#include "VelocityTensorMacros.h"
#include <fftw3.h>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop

namespace mif {
    inline Real compute_eigenvalue_neumann(size_t index, size_t N) {
        return (std::cos(M_PI * index / (N-1)) - 1.0) * (N-1) * (N-1) / (2.0*M_PI*M_PI);
    }

    void solve_pressure_equation_neumann(StaggeredTensor &pressure, StaggeredTensor &pressure_tilde_buffer, const VelocityTensor &velocity, StaggeredTensor &b_buffer, StaggeredTensor &b_tilde_buffer) {
        // TODO: Not tested. Most likely very broken.
        const Constants &constants = velocity.constants;

        // Fill the rhs buffer with the divergence of the velocity.
        VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(pressure, true, b_buffer(i,j,k) = calculate_velocity_divergence(velocity,i,j,k) / constants.dt;)

        // By default, starting from z direction.
        // Create 2decomp object.
        bool neumannBC[3] = {true, true, true};
        C2Decomp c2d = C2Decomp(constants.Nx, constants.Ny, constants.Nz, constants.Pz, constants.Py, neumannBC);


        // Exeucute type 1 DCT along direction x and transpose from (z,y,x) to (x,z,y).
        // Note: major refers to the last index of the triple, the triple is in the iterating order.
        // TODO: only allocate space once.
        Real *temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        Real *temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        fftw_plan fft_plan = fftw_plan_r2r_1d(constants.Nz, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                // Copy the original data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    temp1[i] = b_buffer(i,j,k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    b_tilde_buffer(i,j,k) = temp2[i];
                }
            }
        }
        c2d.transposeX2Y_MajorIndex(static_cast<Real *>(b_tilde_buffer.raw_data()), static_cast<Real *>(b_tilde_buffer.raw_data()));
        b_buffer.swap_data(b_tilde_buffer);

        fftw_free(temp1);
        fftw_free(temp2);


        // Exucute type 1 DCT along direction y and transpose from (x,z,y) to (y,x,z).
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        fft_plan = fftw_plan_r2r_1d(constants.Ny, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                // Copy the original data.
                for (size_t j = 0; j < constants.Ny; j++) {
                    temp1[j] = b_buffer(i,j,k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t j = 0; j < constants.Ny; j++){
                    b_tilde_buffer(i,j,k) = temp2[j];
                }
            }
        }
        c2d.transposeY2Z_MajorIndex(static_cast<Real *>(b_tilde_buffer.raw_data()), static_cast<Real *>(b_tilde_buffer.raw_data()));
        b_buffer.swap_data(b_tilde_buffer);

        fftw_free(temp1);
        fftw_free(temp2);


        // Exucute type 1 DCT along direction z while in indexing (y,x,z), do not transpose.
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        fft_plan = fftw_plan_r2r_1d(constants.Nz, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                // Copy the original data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    temp1[i] = b_buffer(i,j,k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    b_tilde_buffer(i,j,k) = temp2[i];
                }
            }
        }
        b_buffer.swap_data(b_tilde_buffer);

        fftw_free(temp1);
        fftw_free(temp2);


        // Divide by eigenvalues.
        pressure_tilde_buffer.swap_data(b_tilde_buffer);

        for (size_t j = 0; j < constants.Ny; j++) {
            const Real lambda_2 = compute_eigenvalue_neumann(j, constants.Ny);
            for (size_t i = 0; i < constants.Nx; i++) {
                const Real lambda_1= compute_eigenvalue_neumann(i, constants.Nx);
                for (size_t k = 0; k < constants.Nz; k++) {
                    const Real lambda_3 = compute_eigenvalue_neumann(k, constants.Nz);
                    pressure_tilde_buffer(i,j,k) /= (lambda_1 + lambda_2 + lambda_3);
                }
            }
        }
        pressure_tilde_buffer(0,0,0) = 0;


        // Exucute type 1 IDCT along direction z, transpose from (y,x,z) to (x,z,y).
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nz);
        fft_plan = fftw_plan_r2r_1d(constants.Nz, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                // Copy the original data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    temp1[i] = pressure(i,j,k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t k = 0; k < constants.Nz; k++) {
                    pressure_tilde_buffer(i,j,k) = temp2[i] / (2.0*(constants.Nz-1));
                }
            }
        }
        c2d.transposeZ2Y_MajorIndex(static_cast<Real *>(pressure_tilde_buffer.raw_data()), static_cast<Real *>(pressure_tilde_buffer.raw_data()));
        pressure.swap_data(pressure_tilde_buffer);

        fftw_free(temp1);
        fftw_free(temp2);


        // Exucute type 1 IDCT along direction y and transpose from (x,z,y) to (z,y,x).
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Ny);
        fft_plan = fftw_plan_r2r_1d(constants.Ny, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                // Copy the original data.
                for (size_t j = 0; j < constants.Ny; j++) {
                    temp1[j] = pressure(i,j,k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t j = 0; j < constants.Ny; j++){
                    pressure_tilde_buffer(i,j,k) = temp2[j] / (2.0*(constants.Ny-1));
                }
            }
        }
        c2d.transposeY2Z_MajorIndex(static_cast<Real *>(pressure_tilde_buffer.raw_data()), static_cast<Real *>(pressure_tilde_buffer.raw_data()));
        pressure.swap_data(pressure_tilde_buffer);


        // Exucute type 1 IDCT along direction x while in indexing (z,y,x), do not transpose.
        temp1 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        temp2 = (Real*) fftw_malloc(sizeof(Real) * constants.Nx);
        fft_plan = fftw_plan_r2r_1d(constants.Nx, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                // Copy the original data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    temp1[i] = pressure(i,j,k);
                }

                // Execute the fft.
                fftw_execute(fft_plan);

                // Copy the transformed data.
                for (size_t i = 0; i < constants.Nx; i++) {
                    pressure_tilde_buffer(i,j,k) = temp2[i] / (2.0*(constants.Nx-1));
                }
            }
        }
        pressure.swap_data(pressure_tilde_buffer);

        fftw_free(temp1);
        fftw_free(temp2);


        fftw_destroy_plan(fft_plan);
        fftw_cleanup();
    }
}