#include "BoundaryConditions.h"

namespace mif {

    // Apply a Dirichlet boundary condition to the u component of the velocity at the specified point, using the function f.
    inline void apply_dirichlet_bc_u(Tensor &tensor, size_t i, size_t j, size_t k, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        const Real result = f(constants.dx * (i+0.5), constants.dy * j, constants.dz * k);
        tensor.set(i, j, k, result);
    }

    // Apply a Dirichlet boundary condition to the v component of the velocity at the specified point, using the function f.
    inline void apply_dirichlet_bc_v(Tensor &tensor, size_t i, size_t j, size_t k, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        const Real result = f(constants.dx * i, constants.dy * (j+0.5), constants.dz * k);
        tensor.set(i, j, k, result);
    }

    // Apply a Dirichlet boundary condition to the w component of the velocity at the specified point, using the function f.
    inline void apply_dirichlet_bc_w(Tensor &tensor, size_t i, size_t j, size_t k, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        const Real result = f(constants.dx * i, constants.dy * j, constants.dz * (k+0.5));
        tensor.set(i, j, k, result);
    }

    void apply_all_dirichlet_bc(Tensor &u, Tensor &v, Tensor &w, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        // Face 1: z=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                apply_dirichlet_bc_u(u, i, j, 0, f, constants);
                apply_dirichlet_bc_v(v, i, j, 0, f, constants);
                apply_dirichlet_bc_w(w, i, j, 0, f, constants);
            }
        }

        // Face 2: z=z_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                apply_dirichlet_bc_u(u, i, j, constants.Nz-1, f, constants);
                apply_dirichlet_bc_v(v, i, j, constants.Nz-1, f, constants);
                apply_dirichlet_bc_w(w, i, j, constants.Nz-1, f, constants);
            }
        }

        // Face 3: y=0
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc_u(u, i, 0, k, f, constants);
                apply_dirichlet_bc_v(v, i, 0, k, f, constants);
                apply_dirichlet_bc_w(w, i, 0, k, f, constants);
            }
        }

        // Face 4: y=y_max
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc_u(u, i, constants.Ny-1, k, f, constants);
                apply_dirichlet_bc_v(v, i, constants.Ny-1, k, f, constants);
                apply_dirichlet_bc_w(w, i, constants.Ny-1, k, f, constants);
            }
        }

        // Face 5: x=0
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc_u(u, 0, j, k, f, constants);
                apply_dirichlet_bc_v(v, 0, j, k, f, constants);
                apply_dirichlet_bc_w(w, 0, j, k, f, constants);
            }
        }

        // Face 6: x=x_max
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t k = 0; k < constants.Nz; k++) {
                apply_dirichlet_bc_u(u, constants.Nx-1, j, k, f, constants);
                apply_dirichlet_bc_v(v, constants.Nx-1, j, k, f, constants);
                apply_dirichlet_bc_w(w, constants.Nx-1, j, k, f, constants);
            }
        }
    }

} // mif