#include "PressureSolverStructures.h"
#include <cassert>

namespace mif {
    PressureSolverStructures::PressureSolverStructures(const Constants &constants):
        boundary_conditions{true, true, true},
        c2d(C2Decomp(constants.Nx, constants.Ny_global, constants.Nz_global, constants.Py, constants.Pz, boundary_conditions)),
        buffer_x(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Nx))),
        buffer_y(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Ny_global))),
        buffer_z(static_cast<Real*>(fftw_malloc(sizeof(Real) * constants.Nz_global))),
        fft_plan_x(fftw_plan_r2r_1d(constants.Nx, buffer_x, buffer_x, FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_y(fftw_plan_r2r_1d(constants.Ny_global, buffer_y, buffer_y, FFTW_REDFT00, FFTW_ESTIMATE)),
        fft_plan_z(fftw_plan_r2r_1d(constants.Nz_global, buffer_z, buffer_z, FFTW_REDFT00, FFTW_ESTIMATE)) {
        assert(static_cast<size_t>(c2d.xSize[0]) == constants.Nx);
        assert(static_cast<size_t>(c2d.ySize[1]) == constants.Ny_global);
        assert(static_cast<size_t>(c2d.zSize[2]) == constants.Nz_global);
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