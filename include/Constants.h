#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Real.h"

namespace mif {

    class Constants {
        public:
            // Core constants.
            const Real x_size;
            const Real y_size;
            const Real z_size;
            const size_t Nx;
            const size_t Ny;
            const size_t Nz;
            const Real Re;

            // Derived constants (computed here once for efficiency).
            const Real dx;
            const Real dy;
            const Real dz;
            const size_t row_size;
            const size_t matrix_size;
            const Real one_over_2_dx;
            const Real one_over_2_dy;
            const Real one_over_2_dz;
            const Real one_over_8_dx;
            const Real one_over_8_dy;
            const Real one_over_8_dz;
            const Real one_over_dx2_Re;
            const Real one_over_dy2_Re;
            const Real one_over_dz2_Re;

            // Constructor.
            Constants(size_t Nx, size_t Ny, size_t Nz, Real x_size, Real y_size, Real z_size, Real Re);
    }

} // mif

#endif