#include "Constants.h"
#include <cassert>

namespace mif {

    Constants::Constants(size_t Nx, size_t Ny, size_t Nz, Real x_size, Real y_size, Real z_size, Real Re, Real final_time, unsigned int num_time_steps): 
            x_size(x_size), y_size(y_size), z_size(z_size), 
            Nx(Nx), Ny(Ny), Nz(Nz), 
            Re(Re),
            final_time(final_time), num_time_steps(num_time_steps),
            dt(final_time / num_time_steps),
            dx(x_size/(Nx-1)), dy(y_size/(Ny-1)), dz(z_size/(Nz-1)),
            row_size(Nx), matrix_size(Nx*Ny),
            one_over_2_dx(1/(2*dx)), one_over_2_dy(1/(2*dy)), one_over_2_dz(1/(2*dz)),
            one_over_8_dx(1/(8*dx)), one_over_8_dy(1/(8*dy)), one_over_8_dz(1/(8*dz)),
            one_over_dx2_Re(1/(Re*dx*dx)), one_over_dy2_Re(1/(Re*dy*dy)), one_over_dz2_Re(1/(Re*dz*dz)),
            dx_over_2(dx/2), dy_over_2(dy/2), dz_over_2(dz/2),
            one_over_dx(1/dx), one_over_dy(1/dy), one_over_dz(1/dz) {
        assert(Nx > 0 && Ny > 0 && Nz > 0 && num_time_steps > 0 && final_time > 0);
    }

} // mif