#include "PressureSolverStructures.h"
#include <cassert>

namespace mif{
    inline Real compute_eigenvalue_neumann(size_t index, Real delta, size_t N){
        return 2.0 * (std::cos(M_PI * index / (N - 1)) - 1.0) / (delta * delta);
    }

    inline Real compute_eigenvalue_periodic(size_t index, Real delta, size_t N){
        return 2.0 * (std::cos(2.0 * M_PI * index / N) - 1.0) / (delta * delta);
    }

    PressureSolverStructures::PressureSolverStructures(const Constants& constants):
        periodic_bc{constants.periodic_bc[0], constants.periodic_bc[1], constants.periodic_bc[2]},
        Nx_points(constants.Nx_global - (periodic_bc[0] ? 1 : 0)),
        Ny_points(constants.Ny_global - (periodic_bc[1] ? 1 : 0)),
        Nz_points(constants.Nz_global - (periodic_bc[2] ? 1 : 0)),
        c2d(C2Decomp(Nx_points, Ny_points, Nz_points, constants.Py, constants.Pz, periodic_bc)),
        buffer_x(static_cast<Real*>(fftw_malloc(sizeof(Real) * Nx_points))),
        buffer_y(static_cast<Real*>(fftw_malloc(sizeof(Real) * Ny_points))),
        buffer_z(static_cast<Real*>(fftw_malloc(sizeof(Real) * Nz_points))),
#if USE_DOUBLE
        fft_plan_x(fftw_plan_r2r_1d(Nx_points, buffer_x, buffer_x, periodic_bc[0] ? FFTW_R2HC : FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_y(fftw_plan_r2r_1d(Ny_points, buffer_y, buffer_y, periodic_bc[1] ? FFTW_R2HC : FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_z(fftw_plan_r2r_1d(Nz_points, buffer_z, buffer_z, periodic_bc[2] ? FFTW_R2HC : FFTW_REDFT00, FFTW_ESTIMATE)),
        ifft_plan_x(fftw_plan_r2r_1d(Nx_points, buffer_x, buffer_x, periodic_bc[0] ? FFTW_HC2R : FFTW_REDFT00, FFTW_ESTIMATE)),
        ifft_plan_y(fftw_plan_r2r_1d(Ny_points, buffer_y, buffer_y, periodic_bc[1] ? FFTW_HC2R : FFTW_REDFT00, FFTW_ESTIMATE)),
        ifft_plan_z(fftw_plan_r2r_1d(Nz_points, buffer_z, buffer_z, periodic_bc[2] ? FFTW_HC2R : FFTW_REDFT00, FFTW_ESTIMATE)),
#else
        fft_plan_x(fftwf_plan_r2r_1d(Nx_points, buffer_x, buffer_x, periodic_bc[0] ? FFTW_R2HC : FFTW_REDFT00,
                                     FFTW_ESTIMATE)),
        fft_plan_y(fftwf_plan_r2r_1d(Ny_points, buffer_y, buffer_y, periodic_bc[1] ? FFTW_R2HC : FFTW_REDFT00,
                                     FFTW_ESTIMATE)),
        fft_plan_z(fftwf_plan_r2r_1d(Nz_points, buffer_z, buffer_z, periodic_bc[2] ? FFTW_R2HC : FFTW_REDFT00,
                                     FFTW_ESTIMATE)),
        ifft_plan_x(fftwf_plan_r2r_1d(Nx_points, buffer_x, buffer_x, periodic_bc[0] ? FFTW_HC2R : FFTW_REDFT00,
                                      FFTW_ESTIMATE)),
        ifft_plan_y(fftwf_plan_r2r_1d(Ny_points, buffer_y, buffer_y, periodic_bc[1] ? FFTW_HC2R : FFTW_REDFT00,
                                      FFTW_ESTIMATE)),
        ifft_plan_z(fftwf_plan_r2r_1d(Nz_points, buffer_z, buffer_z, periodic_bc[2] ? FFTW_HC2R : FFTW_REDFT00,
                                      FFTW_ESTIMATE)),
#endif

        eigenvalues({c2d.zSize[1], c2d.zSize[0], c2d.zSize[2]}){
        // Check decomposition validity.
        assert(c2d.xSize[0] == Nx_points);
        assert(c2d.ySize[1] == Ny_points);
        assert(c2d.zSize[2] == Nz_points);

        // Initialize eigenvalues.
        assert(c2d.zStart[2] == 0);
        for (int j = 0; j < c2d.zSize[1]; j++){
            const Real lambda_2 = periodic_bc[1]
                                      ? compute_eigenvalue_periodic(j + c2d.zStart[1], constants.dy, Ny_points)
                                      : compute_eigenvalue_neumann(j + c2d.zStart[1], constants.dy, Ny_points);
            const Real base_index_1 = j * c2d.zSize[0] * c2d.zSize[2];
            for (int i = 0; i < c2d.zSize[0]; i++){
                const Real base_index_2 = base_index_1 + i * c2d.zSize[2];
                const Real lambda_1 = periodic_bc[0]
                                          ? compute_eigenvalue_periodic(i + c2d.zStart[0], constants.dx, Nx_points)
                                          : compute_eigenvalue_neumann(i + c2d.zStart[0], constants.dx, Nx_points);
                for (int k = 0; k < c2d.zSize[2]; k++){
                    const Real lambda_3 = periodic_bc[2]
                                              ? compute_eigenvalue_periodic(k, constants.dz, Nz_points)
                                              : compute_eigenvalue_neumann(k, constants.dz, Nz_points);
                    eigenvalues(base_index_2 + k) = 1 / (lambda_1 + lambda_2 + lambda_3);
                }
            }
        }
    }

    PressureSolverStructures::~PressureSolverStructures(){
#if USE_DOUBLE
        fftw_free(buffer_x);
        fftw_free(buffer_y);
        fftw_free(buffer_z);
        fftw_destroy_plan(fft_plan_x);
        fftw_destroy_plan(fft_plan_y);
        fftw_destroy_plan(fft_plan_z);
        fftw_destroy_plan(ifft_plan_x);
        fftw_destroy_plan(ifft_plan_y);
        fftw_destroy_plan(ifft_plan_z);
        fftw_cleanup();
#else
        fftwf_free(buffer_x);
        fftwf_free(buffer_y);
        fftwf_free(buffer_z);
        fftwf_destroy_plan(fft_plan_x);
        fftwf_destroy_plan(fft_plan_y);
        fftwf_destroy_plan(fft_plan_z);
        fftwf_destroy_plan(ifft_plan_x);
        fftwf_destroy_plan(ifft_plan_y);
        fftwf_destroy_plan(ifft_plan_z);
        fftwf_cleanup();


#endif
    }
}
