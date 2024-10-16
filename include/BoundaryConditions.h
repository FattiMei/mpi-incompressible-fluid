#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <functional>
#include "Constants.h"
#include "FunctionHelpers.h"
#include "Tensor.h"
#include "VelocityComponent.h"

namespace mif {

    // Apply a Dirichlet boundary condition to the a component of the velocity on all boundaries.
    // The function assumes the velocity field is divergence free.
    template <VelocityComponent component> void 
    apply_all_dirichlet_bc(Tensor<> &tensor, 
                           const std::function<Real(Real, Real, Real)> &u_exact, 
                           const std::function<Real(Real, Real, Real)> &v_exact, 
                           const std::function<Real(Real, Real, Real)> &w_exact, 
                           const Constants &constants) {
        const std::function<Real(Real, Real, Real)> f = choose_function<component>(u_exact, v_exact, w_exact);

        // Face 1: z=0
        if constexpr (component == VelocityComponent::w) {
            for (size_t i = 1; i < constants.Nx-1; i++) {
                for (size_t j = 1; j < constants.Ny-1; j++) {
                    tensor(i, j, 0) = f(i*constants.dx, j*constants.dy, 0) - constants.dz/2*
                    ((evaluate_staggered<VelocityComponent::u>(i,j,0,u_exact,constants) - evaluate_staggered<VelocityComponent::u>(i-1,j,0,u_exact,constants)) / constants.dx +
                    (evaluate_staggered<VelocityComponent::v>(i,j,0,v_exact,constants) - evaluate_staggered<VelocityComponent::v>(i,j-1,0,v_exact,constants)) / constants.dy);
                }
            }
        } else {
            for (size_t i = 0; i < tensor.sizes()[0]; i++) {
                for (size_t j = 0; j < tensor.sizes()[1]; j++) {
                    tensor(i, j, 0) = evaluate_staggered<component>(i, j, 0, f, constants);
                }
            }
        }

        // Face 2: z=z_max
        if constexpr (component == VelocityComponent::w) {
            for (size_t i = 1; i < constants.Nx-1; i++) {
                for (size_t j = 1; j < constants.Ny-1; j++) {
                    tensor(i, j, constants.Nz-2) = f(i*constants.dx, j*constants.dy, constants.z_size) + constants.dz/2*
                    ((evaluate_staggered<VelocityComponent::u>(i,j,constants.Nz-1,u_exact,constants) - evaluate_staggered<VelocityComponent::u>(i-1,j,constants.Nz-1,u_exact,constants)) / constants.dx +
                    (evaluate_staggered<VelocityComponent::v>(i,j,constants.Nz-1,v_exact,constants) - evaluate_staggered<VelocityComponent::v>(i,j-1,constants.Nz-1,v_exact,constants)) / constants.dy);
                }
            }
        } else {
            for (size_t i = 0; i < tensor.sizes()[0]; i++) {
                for (size_t j = 0; j < tensor.sizes()[1]; j++) {
                    tensor(i, j, constants.Nz-1) = evaluate_staggered<component>(i, j, constants.Nz-1, f, constants);
                }
            }
        }

        // Face 3: y=0
        if constexpr (component == VelocityComponent::v) {
            for (size_t i = 1; i < constants.Nx-1; i++) {
                for (size_t k = 1; k < constants.Nz-1; k++) {
                    tensor(i, 0, k) = f(i*constants.dx, 0, k*constants.dz) - constants.dy/2*
                    ((evaluate_staggered<VelocityComponent::u>(i,0,k,u_exact,constants) - evaluate_staggered<VelocityComponent::u>(i-1,0,k,u_exact,constants)) / constants.dx +
                    (evaluate_staggered<VelocityComponent::w>(i,0,k,w_exact,constants) - evaluate_staggered<VelocityComponent::w>(i,0,k-1,w_exact,constants)) / constants.dz);
                }
            }
        } else {
            for (size_t i = 0; i < tensor.sizes()[0]; i++) {
                for (size_t k = 0; k < tensor.sizes()[2]; k++) {
                    tensor(i, 0, k) = evaluate_staggered<component>(i, 0, k, f, constants);
                }
            }
        }

        // Face 4: y=y_max
        if constexpr (component == VelocityComponent::v) {
            for (size_t i = 1; i < constants.Nx-1; i++) {
                for (size_t k = 1; k < constants.Nz-1; k++) {
                    tensor(i, constants.Ny-2, k) = f(i*constants.dx, constants.y_size, k*constants.dz) + constants.dy/2*
                    ((evaluate_staggered<VelocityComponent::u>(i,constants.Ny-1,k,u_exact,constants) - evaluate_staggered<VelocityComponent::u>(i-1,constants.Ny-1,k,u_exact,constants)) / constants.dx +
                    (evaluate_staggered<VelocityComponent::w>(i,constants.Ny-1,k,w_exact,constants) - evaluate_staggered<VelocityComponent::w>(i,constants.Ny-1,k-1,w_exact,constants)) / constants.dz);
                }
            }
        } else {
            for (size_t i = 0; i < tensor.sizes()[0]; i++) {
                for (size_t k = 0; k < tensor.sizes()[2]; k++) {
                    tensor(i, constants.Ny-1, k) = evaluate_staggered<component>(i, constants.Ny-1, k, f, constants);
                }
            }
        }

        // Face 5: x=0
        if constexpr (component == VelocityComponent::u) {
            for (size_t j = 1; j < constants.Ny-1; j++) {
                for (size_t k = 1; k < constants.Nz-1; k++) {
                    tensor(0, j, k) = f(0, j*constants.dy, k*constants.dz) - constants.dx/2*
                    ((evaluate_staggered<VelocityComponent::v>(0,j,k,v_exact,constants) - evaluate_staggered<VelocityComponent::v>(0,j-1,k,v_exact,constants)) / constants.dy +
                    (evaluate_staggered<VelocityComponent::w>(0,j,k,w_exact,constants) - evaluate_staggered<VelocityComponent::w>(0,j,k-1,w_exact,constants)) / constants.dz);
                }
            }
        } else {
            for (size_t j = 0; j < tensor.sizes()[1]; j++) {
                for (size_t k = 0; k < tensor.sizes()[2]; k++) {
                    tensor(0, j, k) = evaluate_staggered<component>(0, j, k, f, constants);
                }
            }
        }

        // Face 6: x=x_max
        if constexpr (component == VelocityComponent::u) {
            for (size_t j = 1; j < constants.Ny-1; j++) {
                for (size_t k = 1; k < constants.Nz-1; k++) {
                    tensor(constants.Nx-2, j, k) = f(constants.x_size, j*constants.dy, k*constants.dz) + constants.dx/2*
                    ((evaluate_staggered<VelocityComponent::v>(constants.Nx-1,j,k,v_exact,constants) - evaluate_staggered<VelocityComponent::v>(constants.Nx-1,j-1,k,v_exact,constants)) / constants.dy +
                    (evaluate_staggered<VelocityComponent::w>(constants.Nx-1,j,k,w_exact,constants) - evaluate_staggered<VelocityComponent::w>(constants.Nx-1,j,k-1,w_exact,constants)) / constants.dz);
                }
            }
        } else {
            for (size_t j = 0; j < tensor.sizes()[1]; j++) {
                for (size_t k = 0; k < tensor.sizes()[2]; k++) {
                    tensor(constants.Nx-1, j, k) = evaluate_staggered<component>(constants.Nx-1, j, k, f, constants);
                }
            }
        }
    }

} // mif

#endif // BOUNDARY_CONDITIONS_H