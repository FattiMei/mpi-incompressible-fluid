#include "PressureSolverStructures.h"
#include <cassert>

namespace mif {
    inline Real compute_eigenvalue_neumann(size_t index, Real delta, size_t N_domains) {
        return 2.0 * (std::cos(M_PI * index / N_domains) - 1.0) / (delta*delta);
    }
    inline Real compute_eigenvalue_periodic(size_t index, Real delta, size_t N) {
        return 2.0 * (std::cos(2.0 * M_PI * index / N) - 1.0) / (delta*delta);
    }

    PressureSolverStructures::PressureSolverStructures(const Constants &constants):
        periodic_bc{constants.periodic_bc[0], constants.periodic_bc[1], constants.periodic_bc[2]},
        c2d(C2Decomp(constants.Nx, constants.Ny_global, constants.Nz_global, constants.Py, constants.Pz, periodic_bc)),
        buffer_x(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Nx))),
        buffer_y(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Ny_global))),
        buffer_z(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Nz_global))),
        fft_plan_x(fftw_plan_r2r_1d(constants.Nx, buffer_x, buffer_x, FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_y(fftw_plan_r2r_1d(constants.Ny_global, buffer_y, buffer_y, FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_z(fftw_plan_r2r_1d(constants.Nz_global, buffer_z, buffer_z, FFTW_REDFT00, FFTW_ESTIMATE)),
        eigenvalues({c2d.zSize[1], c2d.zSize[0], c2d.zSize[2]}) {
        // Check decomposition validity.
        assert(static_cast<size_t>(c2d.xSize[0]) == constants.Nx);
        assert(static_cast<size_t>(c2d.ySize[1]) == constants.Ny_global);
        assert(static_cast<size_t>(c2d.zSize[2]) == constants.Nz_global);

        // Initialize eigenvalues.
        assert(c2d.zStart[2] == 0);
        for (int j = 0; j < c2d.zSize[1]; j++) {
            const Real lambda_2 = periodic_bc[1] ? 
                compute_eigenvalue_periodic(j+c2d.zStart[1], constants.dy, constants.Ny_global):
                compute_eigenvalue_neumann(j+c2d.zStart[1], constants.dy, constants.Ny_domains_global);
            const Real base_index_1 = j*c2d.zSize[0]*c2d.zSize[2];
            for (int i = 0; i < c2d.zSize[0]; i++) {
                const Real base_index_2 = base_index_1 + i*c2d.zSize[2];
                const Real lambda_1 = periodic_bc[0] ? 
                    compute_eigenvalue_periodic(i+c2d.zStart[0], constants.dx, constants.Nx) :
                    compute_eigenvalue_neumann(i+c2d.zStart[0], constants.dx, constants.Nx_domains);
                for (int k = 0; k < c2d.zSize[2]; k++) {
                    const Real lambda_3 = periodic_bc[2] ? 
                        compute_eigenvalue_periodic(k, constants.dz, constants.Nz_global) :
                        compute_eigenvalue_neumann(k, constants.dz, constants.Nz_domains_global);
                    eigenvalues(base_index_2 + k) = 1/(lambda_1 + lambda_2 + lambda_3);
                }
            }
        }
    }
    
    PressureSolverStructures::~PressureSolverStructures() {
        fftw_free(buffer_x);
        fftw_free(buffer_y);
        fftw_free(buffer_z);
        fftw_destroy_plan(fft_plan_x);
        fftw_destroy_plan(fft_plan_y);
        fftw_destroy_plan(fft_plan_z);
        fftw_cleanup();
    }
}