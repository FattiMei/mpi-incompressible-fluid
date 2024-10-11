#include "InitialConditions.h"

namespace mif {

    void apply_initial_conditions_u(Tensor &u, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                for (size_t k = 0; k < constants.Nz; k++) {
                    const Real result = f(constants.dx * (i+0.5), constants.dy * j, constants.dz * k);
                    u.set(i, j, k, result);
                }
            }
        }
    }

    void apply_initial_conditions_v(Tensor &v, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                for (size_t k = 0; k < constants.Nz; k++) {
                    const Real result = f(constants.dx * i, constants.dy * (j+0.5), constants.dz * k);
                    v.set(i, j, k, result);
                }
            }
        }
    }

    void apply_initial_conditions_w(Tensor &w, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        for (size_t i = 0; i < constants.Nx; i++) {
            for (size_t j = 0; j < constants.Ny; j++) {
                for (size_t k = 0; k < constants.Nz; k++) {
                    const Real result = f(constants.dx * i, constants.dy * j, constants.dz * (k+0.5));
                    w.set(i, j, k, result);
                }
            }
        }
    }

} // mif