//
// Created by giorgio on 10/10/2024.
//

#include <functional>
#include "Constants.h"
#include "MomentumEquation.h"
#include "Tensor.h"
#include "Timestep.h"

namespace mif {
    // Perform a single step of an explicit RK3 method for a single point.
    Real RK3(Real x, Real dt, const std::function<Real(size_t, size_t, size_t, Real)> &f, size_t i, size_t j, size_t k);

    void timestep(Tensor &u, Tensor &v, Tensor &w, const Constants &constants) {
        auto f_u = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            u.set(i, j, k, x);
            return calculate_momentum_rhs_u(u, v, w, i, j, k, constants);
        };

        auto f_v = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            v.set(i, j, k, x);
            return calculate_momentum_rhs_v(u, v, w, i, j, k, constants);
        };
        auto f_w = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            w.set(i, j, k, x);
            return calculate_momentum_rhs_w(u, v, w, i, j, k, constants);
        };

        // Update the velocity solution inside the mesh.
        for (size_t i = 1; i < u.size[0] - 1; i++) {
            for (size_t j = 1; j < u.size[1] - 1; j++) {
                for (size_t k = 1; k < u.size[2] - 1; k++) {
                    // Update the solution. I don't know if updating each component separately is the correct way to do it since the components are dependent on each other.
                    u.set(i, j, k, RK3(u.get(i, j, k), constants.dt, f_u, i, j, k));
                    v.set(i, j, k, RK3(v.get(i, j, k), constants.dt, f_v, i, j, k));
                    w.set(i, j, k, RK3(w.get(i, j, k), constants.dt, f_w, i, j, k));
                }
            }
        }
    }


    Real
    RK3(Real x, Real dt, const std::function<Real(size_t, size_t, size_t, Real)> &f, size_t i, size_t j, size_t k) {
        constexpr Real a1 = (64.0 / 120.0);
        constexpr Real a2 = (50.0 / 120.0);
        constexpr Real a3 = (-34.0 / 120.0);
        constexpr Real a4 = (90.0 / 120.0);
        constexpr Real a5 = (-50.0 / 120.0);
        // Y2 = u + ... Step
        const Real Y2 = x + dt * f(i, j, k, x) * a1;
        // Y3 = Y2 + ... Step
        const Real Y3 = Y2 + dt * a2 * f(i, j, k, Y2) + dt * a3 * f(i, j, k, x);
        // Y3 = Y2 + ... Step
        x = Y2 + dt * a5 * f(i, j, k, x) + dt * a4 * f(i, j, k, Y3);
        return x;
    }
}