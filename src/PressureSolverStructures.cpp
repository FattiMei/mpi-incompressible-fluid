#include "PressureSolverStructures.h"

namespace mif {
    PressureSolverStructures::PressureSolverStructures(const Constants &constants):
        boundary_conditions{true, true, true},
        c2d(C2Decomp(constants.Nx, constants.Ny_domains_global+2*constants.Py-1, constants.Nz_domains_global+2*constants.Pz-1, constants.Py, constants.Pz, boundary_conditions)),
        buffer_x(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Nx))),
        buffer_y(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Ny))),
        buffer_z(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Nz))),
        fft_plan_x(fftw_plan_r2r_1d(constants.Nx, buffer_x, buffer_x, FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_y(fftw_plan_r2r_1d(constants.Ny, buffer_y, buffer_y, FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_z(fftw_plan_r2r_1d(constants.Nz, buffer_z, buffer_z, FFTW_REDFT00, FFTW_ESTIMATE)) {}
    
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