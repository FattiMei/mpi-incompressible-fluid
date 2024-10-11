#include "InitialConditions.h"

namespace mif {

    void apply_initial_conditions(Tensor &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                for (size_t k = 0; k < constants.Nz; k++) {
                    const Real result = f(constants.dx * i, constants.dy * j, constants.dz * k);
                    tensor.set(i, j, k, result);
                }
            }
        }
    }

} // mif