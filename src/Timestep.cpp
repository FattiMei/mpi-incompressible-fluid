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

    std::array<Real, 3>
    RK3_coupled(Real u_n, Real v_n, Real w_n, Real dt, const std::function<Real(size_t, size_t, size_t, Real)> &f_u,
                const std::function<Real(size_t, size_t, size_t, Real)> &f_v,
                const std::function<Real(size_t, size_t, size_t, Real)> &f_w, size_t i, size_t j, size_t k);

    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, const Constants &constants) {
        auto f_u = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            u(i, j, k) = x;
            return calculate_momentum_rhs_u(u, v, w, i, j, k, constants);
        };

        auto f_v = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            v(i, j, k) = x;
            return calculate_momentum_rhs_v(u, v, w, i, j, k, constants);
        };
        auto f_w = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            w(i, j, k) = x;
            return calculate_momentum_rhs_w(u, v, w, i, j, k, constants);
        };

        // Update the velocity solution inside the mesh.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    // Update the solution. I don't know if updating each component separately is the correct way to do it since the components are dependent on each other.
                    /* u.set(i, j, k, RK3(u.get(i, j, k), constants.dt, f_u, i, j, k));
                     v.set(i, j, k, RK3(v.get(i, j, k), constants.dt, f_v, i, j, k));
                     w.set(i, j, k, RK3(w.get(i, j, k), constants.dt, f_w, i, j, k));
                     */
                    //Probably this is better for stability reasons, but we would need to check it
                    auto result = RK3_coupled(u(i, j, k), v(i, j, k), w(i, j, k), constants.dt, f_u, f_v,
                                              f_w, i, j, k);
                    u(i, j, k) = result[0];
                    v(i, j, k) = result[1];
                    w(i, j, k) = result[2];
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

    std::array<Real, 3> RK3_coupled(double u_n, double v_n, double w_n, double dt,
                                    const std::function<double(size_t, size_t, size_t, double)> &f_u,
                                    const std::function<double(size_t, size_t, size_t, double)> &f_v,
                                    const std::function<double(size_t, size_t, size_t, double)> &f_w, size_t i,
                                    size_t j, size_t k) {
        constexpr Real a1 = (64.0 / 120.0);
        constexpr Real a2 = (50.0 / 120.0);
        constexpr Real a3 = (-34.0 / 120.0);
        constexpr Real a4 = (90.0 / 120.0);
        constexpr Real a5 = (-50.0 / 120.0);
        // Y2 = u + ... Step
        const Real Y2_u = u_n + dt * f_u(i, j, k, u_n) * a1;
        const Real Y2_v = v_n + dt * f_v(i, j, k, v_n) * a1;
        const Real Y2_w = w_n + dt * f_w(i, j, k, w_n) * a1;
        // Y3 = Y2 + ... Step
        const Real Y3_u = Y2_u + dt * a2 * f_u(i, j, k, Y2_u) + dt * a3 * f_u(i, j, k, u_n);
        const Real Y3_v = Y2_v + dt * a2 * f_v(i, j, k, Y2_v) + dt * a3 * f_v(i, j, k, v_n);
        const Real Y3_w = Y2_w + dt * a2 * f_w(i, j, k, Y2_w) + dt * a3 * f_w(i, j, k, w_n);
        // Y3 = Y2 + ... Step
        u_n = Y2_u + dt * a5 * f_u(i, j, k, u_n) + dt * a4 * f_u(i, j, k, Y3_u);
        v_n = Y2_v + dt * a5 * f_v(i, j, k, v_n) + dt * a4 * f_v(i, j, k, Y3_v);
        w_n = Y2_w + dt * a5 * f_w(i, j, k, w_n) + dt * a4 * f_w(i, j, k, Y3_w);

        return {u_n, v_n, w_n};
    }

}