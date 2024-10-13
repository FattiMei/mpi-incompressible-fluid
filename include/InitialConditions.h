#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include "Constants.h"
#include "FunctionHelpers.h"
#include "Tensor.h"
#include "VelocityComponent.h"

namespace mif {

    // Apply initial conditions to all points of the a component of the velocty, using the function f.
    template <VelocityComponent component> void 
    apply_initial_conditions(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                for (size_t k = 0; k < constants.Nz; k++) {
                    tensor(i, j, k) = evaluate_staggered<component>(tensor, i, j, k, f, constants);
                }
            }
        }
    }

} // mif

#endif // INITIAL_CONDITIONS_H