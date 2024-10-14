//
// Created by giorgio on 10/10/2024.
//

#include <cassert>
#include "BoundaryConditions.h"
#include "Constants.h"
#include "MomentumEquation.h"
#include "FunctionHelpers.h"
#include "Tensor.h"
#include "Timestep.h"
#include "VelocityComponent.h"

namespace mif {
    //constexpr Real a1 = (64.0 / 120.0);
    //constexpr Real a2 = (50.0 / 120.0);
    //constexpr Real a3 = (-34.0 / 120.0);
    //constexpr Real a4 = (90.0 / 120.0);
    //constexpr Real a5 = (-50.0 / 120.0);
    //constexpr Real partial_time_1 = (64.0 / 120.0);
    //constexpr Real partial_time_2 = (80.0 / 120.0);
    constexpr Real c2 = (8.0 / 15.0);
    constexpr Real a21 = (8.0 / 15.0);
    constexpr Real c3 = (2.0 / 3.0);
    constexpr Real a31 = (1.0 / 4.0);
    constexpr Real a32 = (5.0 / 12.0);
    constexpr Real b1 = (1.0 / 4.0);
    constexpr Real b3 = (3.0 / 4.0);

    // Perform the first stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_Y2(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w, 
               const std::function<Real(Real, Real, Real, Real)> &forcing_term,
               Real current_time, const Constants &constants, 
               size_t i, size_t j, size_t k) {
        const Real initial_term = choose_component<component>(u, v, w)(i,j,k);
        return initial_term + 
               constants.dt * a21 * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, function_at_time(forcing_term, current_time), constants);
    }

    // Perform the second stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_Y3(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w, 
               const Tensor<> &Y2_u, const Tensor<> &Y2_v, const Tensor<> &Y2_w,
               const std::function<Real(Real, Real, Real, Real)> &forcing_term,
               Real current_time, const Constants &constants, 
               size_t i, size_t j, size_t k) {
        const Real time_1 = current_time + c2*constants.dt;
        const Real initial_term = choose_component<component>(u, v, w)(i,j,k);
        return initial_term + 
               constants.dt * a31 * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, function_at_time(forcing_term, current_time), constants) +
               constants.dt * a32 * calculate_momentum_rhs_with_forcing<component>(Y2_u, Y2_v, Y2_w, i, j, k, function_at_time(forcing_term, time_1), constants);
    }

    // Perform the third stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_new_u(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w,
                  const Tensor<> &Y3_u, const Tensor<> &Y3_v, const Tensor<> &Y3_w,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term,
                  Real current_time, const Constants &constants, 
                  size_t i, size_t j, size_t k) {
        const Real time_2 = current_time + c3*constants.dt;
        const Real initial_term = choose_component<component>(u, v, w)(i,j,k);
        return initial_term + 
               constants.dt * b1 * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, function_at_time(forcing_term, current_time), constants) + 
               constants.dt * b3 * calculate_momentum_rhs_with_forcing<component>(Y3_u, Y3_v, Y3_w, i, j, k, function_at_time(forcing_term, time_2), constants);
    }

    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, 
                  Tensor<> &u_buffer1, Tensor<> &v_buffer1, Tensor<> &w_buffer1,
                  Tensor<> &u_buffer2, Tensor<> &v_buffer2, Tensor<> &w_buffer2, 
                  const std::function<Real(Real, Real, Real, Real)> &u_exact,
                  const std::function<Real(Real, Real, Real, Real)> &v_exact,
                  const std::function<Real(Real, Real, Real, Real)> &w_exact, 
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_u,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_v,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_w,
                  Real current_time,
                  const Constants &constants) {
        const Real time_1 = current_time + c2*constants.dt;
        const Real time_2 = current_time + c3*constants.dt;

        // Stage 1.
        // Apply Dirichlet boundary conditions.
        apply_all_dirichlet_bc<VelocityComponent::u>(u_buffer1, function_at_time(u_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v_buffer1, function_at_time(v_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w_buffer1, function_at_time(w_exact, current_time), constants);

        // Compute the solution inside the domain.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    u_buffer1(i, j, k) = compute_Y2<VelocityComponent::u>(u, v, w, forcing_term_u, current_time, constants, i, j, k);
                    v_buffer1(i, j, k) = compute_Y2<VelocityComponent::v>(u, v, w, forcing_term_v, current_time, constants, i, j, k);
                    w_buffer1(i, j, k) = compute_Y2<VelocityComponent::w>(u, v, w, forcing_term_w, current_time, constants, i, j, k);
                }
            }
        }

        // Stage 2.
        // Apply Dirichlet boundary conditions.
        apply_all_dirichlet_bc<VelocityComponent::u>(u_buffer2, function_at_time(u_exact, time_1), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v_buffer2, function_at_time(v_exact, time_1), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w_buffer2, function_at_time(w_exact, time_1), constants);

        // Compute the solution inside the domain.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    u_buffer2(i, j, k) = compute_Y3<VelocityComponent::u>(u, v, w, u_buffer1, v_buffer1, w_buffer1, forcing_term_u, current_time, constants, i, j, k);
                    v_buffer2(i, j, k) = compute_Y3<VelocityComponent::v>(u, v, w, u_buffer1, v_buffer1, w_buffer1, forcing_term_v, current_time, constants, i, j, k);
                    w_buffer2(i, j, k) = compute_Y3<VelocityComponent::w>(u, v, w, u_buffer1, v_buffer1, w_buffer1, forcing_term_w, current_time, constants, i, j, k);
                }
            }
        }

        // Stage 3.
        // Apply Dirichlet boundary conditions.
        apply_all_dirichlet_bc<VelocityComponent::u>(u_buffer1, function_at_time(u_exact, time_2), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v_buffer1, function_at_time(v_exact, time_2), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w_buffer1, function_at_time(w_exact, time_2), constants);

        // Compute the solution inside the domain.
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    u_buffer1(i, j, k) = compute_new_u<VelocityComponent::u>(u, v, w, u_buffer2, v_buffer2, w_buffer2, forcing_term_u, current_time, constants, i, j, k);
                    v_buffer1(i, j, k) = compute_new_u<VelocityComponent::v>(u, v, w, u_buffer2, v_buffer2, w_buffer2, forcing_term_v, current_time, constants, i, j, k);
                    w_buffer1(i, j, k) = compute_new_u<VelocityComponent::w>(u, v, w, u_buffer2, v_buffer2, w_buffer2, forcing_term_w, current_time, constants, i, j, k);
                }
            }
        }

        // Put the solution in the original tensors.
        u.swap_data(u_buffer1);
        v.swap_data(v_buffer1);
        w.swap_data(w_buffer1);
    }
}
