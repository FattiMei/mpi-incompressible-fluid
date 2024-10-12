#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <functional>
#include "Constants.h"
#include "Tensor.h"
#include "VelocityComponent.h"

namespace mif {

    // Apply a Dirichlet boundary condition to the u component of the velocity at the specified point, using the function f.
    template <VelocityComponent component> inline void 
    apply_dirichlet_bc(Tensor<> &tensor, size_t i, size_t j, size_t k, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        switch (component)
        {
        case VelocityComponent::u: {
                tensor(i, j, k) = f(constants.dx * (i+0.5), constants.dy * j, constants.dz * k);
                return;
            }
        case VelocityComponent::v: {
                tensor(i, j, k) = f(constants.dx * i, constants.dy * (j+0.5), constants.dz * k);
                return;
            }
        case VelocityComponent::w: {
                tensor(i, j, k) = f(constants.dx * i, constants.dy * j, constants.dz * (k+0.5));
                return;
            }
        }
    }

    template <VelocityComponent component> void 
    apply_all_dirichlet_bc(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        // Face 1: z=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                apply_dirichlet_bc<component>(tensor, i, j, 0, f, constants);
            }
        }

        // Face 2: z=z_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                apply_dirichlet_bc<component>(tensor, i, j, constants.Nz-1, f, constants);
            }
        }

        // Face 3: y=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc<component>(tensor, i, 0, k, f, constants);
            }
        }

        // Face 4: y=y_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc<component>(tensor, i, constants.Ny-1, k, f, constants);
            }
        }

        // Face 5: x=0
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc<component>(tensor, 0, j, k, f, constants);
            }
        }

        // Face 6: x=x_max
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc<component>(tensor, 0, j, constants.Nx-1, f, constants);
            }
        }
    }

} // mif

#endif // BOUNDARY_CONDITIONS_H