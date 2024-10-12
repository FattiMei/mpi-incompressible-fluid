//
// Created by giorgio on 10/10/2024.
//

#include <functional>
#include "Constants.h"
#include "MomentumEquation.h"
#include "Tensor.h"
#include "Timestep.h"

namespace mif {

    // Perform the first stage of a single step of an explicit RK3 method for a single point of a single component.
    inline Real
    compute_Y2(const Tensor<> &u, Real dt, const std::function<Real(size_t, size_t, size_t, Real)> &f, size_t i, size_t j, size_t k) {
        constexpr Real a1 = (64.0 / 120.0);
        return u(i,j,k) + dt * f(i, j, k, u(i,j,k)) * a1;
    }

    // Perform the second stage of a single step of an explicit RK3 method for a single point of a single component.
    inline Real 
    compute_Y3(const Tensor<> &u, const Tensor<> &Y2, Real dt, const std::function<Real(size_t, size_t, size_t, Real)> &f, size_t i, size_t j, size_t k) {
        constexpr Real a2 = (50.0 / 120.0);
        constexpr Real a3 = (-34.0 / 120.0);
        return Y2(i,j,k) + dt * a2 * f(i, j, k, Y2(i,j,k)) + dt * a3 * f(i, j, k, u(i,j,k));
    }

    // Perform the third stage of a single step of an explicit RK3 method for a single point of a single component.
    inline Real 
    compute_new_u(const Tensor<> &u, const Tensor<> &Y2, const Tensor<> &Y3, Real dt, const std::function<Real(size_t, size_t, size_t, Real)> &f, size_t i, size_t j, size_t k) {
        constexpr Real a4 = (90.0 / 120.0);
        constexpr Real a5 = (-50.0 / 120.0);
        return Y2(i,j,k) + dt * a5 * f(i, j, k, u(i,j,k)) + dt * a4 * f(i, j, k, Y3(i,j,k));
    }

    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, 
                  Tensor<> &u_buffer1, Tensor<> &v_buffer1, Tensor<> &w_buffer1,
                  Tensor<> &u_buffer2, Tensor<> &v_buffer2, Tensor<> &w_buffer2,  
                  Tensor<> &u_buffer3, Tensor<> &v_buffer3, Tensor<> &w_buffer3,  
                  const Constants &constants) {
        auto f_u = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            return calculate_momentum_rhs_u(u, v, w, i, j, k, constants);
        };
        auto f_v = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            return calculate_momentum_rhs_v(u, v, w, i, j, k, constants);
        };
        auto f_w = [&u, &v, &w, &constants](size_t i, size_t j, size_t k, Real x) {
            return calculate_momentum_rhs_w(u, v, w, i, j, k, constants);
        };

        // Update the velocity solution inside the mesh - stage 1.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    u_buffer1(i, j, k) = compute_Y2(u, constants.dt, f_u, i, j, k);
                    v_buffer1(i, j, k) = compute_Y2(v, constants.dt, f_v, i, j, k);
                    w_buffer1(i, j, k) = compute_Y2(w, constants.dt, f_w, i, j, k);
                }
            }
        }

        // Update the velocity solution inside the mesh - stage 2.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    u_buffer2(i, j, k) = compute_Y3(u, u_buffer1, constants.dt, f_u, i, j, k);
                    v_buffer2(i, j, k) = compute_Y3(v, v_buffer1, constants.dt, f_v, i, j, k);
                    w_buffer2(i, j, k) = compute_Y3(w, w_buffer1, constants.dt, f_w, i, j, k);
                }
            }
        }

        // Update the velocity solution inside the mesh - stage 3.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    u_buffer3(i, j, k) = compute_new_u(u, u_buffer1, u_buffer2, constants.dt, f_u, i, j, k);
                    v_buffer3(i, j, k) = compute_new_u(v, v_buffer1, v_buffer2, constants.dt, f_v, i, j, k);
                    w_buffer3(i, j, k) = compute_new_u(w, w_buffer1, w_buffer2, constants.dt, f_w, i, j, k);
                }
            }
        }

        // Insert the new solution into the original tensors.
        u.swap_data(u_buffer3);
        v.swap_data(v_buffer3);
        w.swap_data(w_buffer3);
    }
}