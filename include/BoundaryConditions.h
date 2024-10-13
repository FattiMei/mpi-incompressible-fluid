#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <functional>
#include "Constants.h"
#include "Tensor.h"
#include "VelocityComponent.h"

namespace mif {

    // Apply a Dirichlet boundary condition to the a component of the velocity on all boundaries, using the function f.
    template <VelocityComponent component> void 
    apply_all_dirichlet_bc(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        // Face 1: z=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                tensor(i, j, 0) = evaluate_staggered<component>(tensor, i, j, 0, f, constants);
            }
        }

        // Face 2: z=z_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                tensor(i, j, constants.Nz-1) = evaluate_staggered<component>(tensor, i, j, constants.Nz-1, f, constants);
            }
        }

        // Face 3: y=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                tensor(i, 0, k) = evaluate_staggered<component>(tensor, i, 0, k, f, constants);
            }
        }

        // Face 4: y=y_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                tensor(i, constants.Ny-1, k) = evaluate_staggered<component>(tensor, i, constants.Ny-1, k, f, constants);
            }
        }

        // Face 5: x=0
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                tensor(0, j, k) = evaluate_staggered<component>(tensor, 0, j, k, f, constants);
            }
        }

        // Face 6: x=x_max
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                tensor(constants.Nx-1, j, k) = evaluate_staggered<component>(tensor, constants.Nx-1, j, k, f, constants);
            }
        }
    }

} // mif

#endif // BOUNDARY_CONDITIONS_H