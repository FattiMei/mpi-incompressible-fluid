#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <functional>
#include "Constants.h"
#include "Tensor.h"
#include "VelocityComponent.h"

namespace mif {

    // Apply initial conditions to all points of the a component of the velocty, using the function f.
    template <VelocityComponent component> void 
    apply_initial_conditions(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        for (size_t i = 0; i < constants.Nx; i++) {
            Real new_i = i;
            if constexpr(component == 0) {
                new_i += 0.5;
            }

            for (size_t j = 0; j < constants.Ny; j++) {
                Real new_j = j;
                if constexpr(component == 1) {
                    new_j += 0.5;
                }

                for (size_t k = 0; k < constants.Nz; k++) {
                    Real new_k = k;
                    if constexpr(component == 2) {
                        new_i += 0.5;
                    }

                    const Real result = f(constants.dx * new_i, constants.dy * new_j, constants.dz * new_k);
                    tensor(i, j, k) = result;
                }
            }
        }
    }

} // mif

#endif // INITIAL_CONDITIONS_H