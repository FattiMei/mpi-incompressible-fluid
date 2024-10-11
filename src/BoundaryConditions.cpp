#include "BoundaryConditions.h"

namespace mif {

    // Apply a Dirichlet boundary condition to the input tensor at the specified point, using the function f.
    inline void apply_dirichlet_bc(Tensor &tensor, size_t i, size_t j, size_t k, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        const Real result = f(constants.dx * i, constants.dy * j, constants.dz * k);
        tensor.set(i, j, k, result);
    }

    void apply_all_dirichlet_bc(Tensor &u, Tensor &v, Tensor &w, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        // Face 1: z=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                apply_dirichlet_bc(u, i, j, 0, f, constants);
                apply_dirichlet_bc(v, i, j, 0, f, constants);
                apply_dirichlet_bc(w, i, j, 0, f, constants);
            }
        }

        // Face 2: z=z_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                apply_dirichlet_bc(u, i, j, constants.Nz-1, f, constants);
                apply_dirichlet_bc(v, i, j, constants.Nz-1, f, constants);
                apply_dirichlet_bc(w, i, j, constants.Nz-1, f, constants);
            }
        }

        // Face 3: y=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc(u, i, 0, k, f, constants);
                apply_dirichlet_bc(v, i, 0, k, f, constants);
                apply_dirichlet_bc(w, i, 0, k, f, constants);
            }
        }

        // Face 4: y=y_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc(u, i, constants.Ny-1, k, f, constants);
                apply_dirichlet_bc(v, i, constants.Ny-1, k, f, constants);
                apply_dirichlet_bc(w, i, constants.Ny-1, k, f, constants);
            }
        }

        // Face 5: x=0
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc(u, 0, j, k, f, constants);
                apply_dirichlet_bc(v, 0, j, k, f, constants);
                apply_dirichlet_bc(w, 0, j, k, f, constants);
            }
        }

        // Face 6: x=x_max
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc(u, constants.Nx-1, j, k, f, constants);
                apply_dirichlet_bc(v, constants.Nx-1, j, k, f, constants);
                apply_dirichlet_bc(w, constants.Nx-1, j, k, f, constants);
            }
        }
    }

} // mif