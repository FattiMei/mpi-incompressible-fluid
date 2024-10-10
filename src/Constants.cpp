#include <cassert>
#include "Constants.h"

namespace mif {

    Constants::Constants(size_t Nx, size_t Ny, size_t Nz, Real x_size, Real y_size, Real z_size, Real Re): 
            Nx(Nx), Ny(Ny), Nz(Nz), 
            x_size(x_size), y_size(y_size), z_size(z_size), 
            Re(Re),
            dx(x_size/Nx), dy(y_size/Ny), dz(z_size/Nz),
            row_size(Nx), matrix_size(Nx*Ny),
            one_over_2_dx(1/(2*dx)), one_over_2_dy(1/(2*dy)), one_over_2_dz(1/(2*dz)),
            one_over_8_dx(1/(8*dx)), one_over_8_dy(1/(8*dy)), one_over_8_dz(1/(8*dz)),
            one_over_dx2_Re(1/(Re*dx*dx)), one_over_dy2_Re(1/(Re*dy*dy)), one_over_dz2_Re(1/(Re*dz*dz)) {
        assert(Nx > 0 && Ny > 0 && Nz > 0);
    }

} // mif