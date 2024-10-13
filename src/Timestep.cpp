//
// Created by giorgio on 10/10/2024.
//

#include <cassert>
#include <functional>
#include "Constants.h"
#include "MomentumEquation.h"
#include "Tensor.h"
#include "Timestep.h"
#include "VelocityComponent.h"

namespace mif {

    // Perform the first stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_Y2(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w, 
               const std::function<Real(Real, Real, Real)> &forcing_term,
               const Constants &constants, size_t i, size_t j, size_t k) {
        constexpr Real a1 = (64.0 / 120.0);
        Real initial_term;
        if constexpr (component == VelocityComponent::u) {
            initial_term = u(i,j,k);
        } else if constexpr (component == VelocityComponent::v) {
            initial_term = v(i,j,k);
        } else {
            static_assert(component == VelocityComponent::w);
            initial_term = w(i,j,k);
        }
        return initial_term + 
               constants.dt * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, forcing_term, constants) * a1;
    }

    // Perform the second stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_Y3(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w, 
               const Tensor<> &Y2_u, const Tensor<> &Y2_v, const Tensor<> &Y2_w,
               const std::function<Real(Real, Real, Real)> &forcing_term,
               const Constants &constants, size_t i, size_t j, size_t k) {
        constexpr Real a2 = (50.0 / 120.0);
        constexpr Real a3 = (-34.0 / 120.0);
        Real initial_term;
        if constexpr (component == VelocityComponent::u) {
            initial_term = Y2_u(i,j,k);
        } else if constexpr (component == VelocityComponent::v) {
            initial_term = Y2_v(i,j,k);
        } else {
            static_assert(component == VelocityComponent::w);
            initial_term = Y2_w(i,j,k);
        }
        return initial_term + 
               constants.dt * a2 * calculate_momentum_rhs_with_forcing<component>(Y2_u, Y2_v, Y2_w, i, j, k, forcing_term, constants) + 
               constants.dt * a3 * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, forcing_term, constants);
    }

    // Perform the third stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_new_u(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w, 
               const Tensor<> &Y2_u, const Tensor<> &Y2_v, const Tensor<> &Y2_w,
               const Tensor<> &Y3_u, const Tensor<> &Y3_v, const Tensor<> &Y3_w,
               const std::function<Real(Real, Real, Real)> &forcing_term,
               const Constants &constants, size_t i, size_t j, size_t k) {
        constexpr Real a4 = (90.0 / 120.0);
        constexpr Real a5 = (-50.0 / 120.0);
        Real initial_term;
        if constexpr (component == VelocityComponent::u) {
            initial_term = Y2_u(i,j,k);
        } else if constexpr (component == VelocityComponent::v) {
            initial_term = Y2_v(i,j,k);
        } else {
            static_assert(component == VelocityComponent::w);
            initial_term = Y2_w(i,j,k);
        }
        return initial_term + 
               constants.dt * a5 * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, forcing_term, constants) + 
               constants.dt * a4 * calculate_momentum_rhs_with_forcing<component>(Y3_u, Y3_v, Y3_w, i, j, k, forcing_term, constants);
    }

    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, 
                  Tensor<> &u_buffer1, Tensor<> &v_buffer1, Tensor<> &w_buffer1,
                  Tensor<> &u_buffer2, Tensor<> &v_buffer2, Tensor<> &w_buffer2,  
                  Tensor<> &u_buffer3, Tensor<> &v_buffer3, Tensor<> &w_buffer3,  
                  const std::function<Real(Real, Real, Real)> &forcing_term_u,
                  const std::function<Real(Real, Real, Real)> &forcing_term_v,
                  const std::function<Real(Real, Real, Real)> &forcing_term_w,
                  const Constants &constants) {
        // Update the velocity solution inside the mesh.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    // Stage 1.
                    u_buffer1(i, j, k) = compute_Y2<VelocityComponent::u>(u, v, w, forcing_term_u, constants, i, j, k);
                    v_buffer1(i, j, k) = compute_Y2<VelocityComponent::v>(u, v, w, forcing_term_v, constants, i, j, k);
                    w_buffer1(i, j, k) = compute_Y2<VelocityComponent::w>(u, v, w, forcing_term_w, constants, i, j, k);

                    // Stage 2.
                    u_buffer2(i, j, k) = compute_Y3<VelocityComponent::u>(u, v, w, u_buffer1, v_buffer1, w_buffer1, forcing_term_u, constants, i, j, k);
                    v_buffer2(i, j, k) = compute_Y3<VelocityComponent::v>(u, v, w, u_buffer1, v_buffer1, w_buffer1, forcing_term_v, constants, i, j, k);
                    w_buffer2(i, j, k) = compute_Y3<VelocityComponent::w>(u, v, w, u_buffer1, v_buffer1, w_buffer1, forcing_term_w, constants, i, j, k);

                    // Stage 3.
                    u_buffer3(i, j, k) = compute_new_u<VelocityComponent::u>(u, v, w, u_buffer1, v_buffer1, w_buffer1, u_buffer2, v_buffer2, w_buffer2, forcing_term_u, constants, i, j, k);
                    v_buffer3(i, j, k) = compute_new_u<VelocityComponent::v>(u, v, w, u_buffer1, v_buffer1, w_buffer1, u_buffer2, v_buffer2, w_buffer2, forcing_term_v, constants, i, j, k);
                    w_buffer3(i, j, k) = compute_new_u<VelocityComponent::w>(u, v, w, u_buffer1, v_buffer1, w_buffer1, u_buffer2, v_buffer2, w_buffer2, forcing_term_w, constants, i, j, k);
                }
            }
        }

        // Insert the new solution into the original tensors.
        u.swap_data(u_buffer3);
        v.swap_data(v_buffer3);
        w.swap_data(w_buffer3);
    }
}
