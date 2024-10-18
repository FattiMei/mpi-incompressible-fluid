//
// Created by giorgio on 10/10/2024.
//

#include "Timestep.h"
#include "MomentumEquation.h"

namespace mif {

    constexpr Real c2 = (8.0 / 15.0);
    constexpr Real a21 = (8.0 / 15.0);
    constexpr Real c3 = (2.0 / 3.0);
    constexpr Real a31 = (1.0 / 4.0);
    constexpr Real a32 = (5.0 / 12.0);
    constexpr Real b1 = (1.0 / 4.0);
    constexpr Real b3 = (3.0 / 4.0);

    // Perform the first stage of a single step of an explicit RK3 method for a
    // single point of a single component.
    #define COMPUTE_Y2(component, equation)                               \
        const Real dt = U.constants.dt;                                   \
        return U.component(i,j,k) + dt * a21 * equation(U, i, j, k, t_n); \
        
    inline Real compute_Y2_u(const VelocityTensor &U, Real t_n,
                             size_t i, size_t j, size_t k) {
        COMPUTE_Y2(u, calculate_momentum_rhs_with_forcing_u);
    }

    inline Real compute_Y2_v(const VelocityTensor &U, Real t_n,
                             size_t i, size_t j, size_t k) {
        COMPUTE_Y2(v, calculate_momentum_rhs_with_forcing_v);
    }

    inline Real compute_Y2_w(const VelocityTensor &U, Real t_n,
                             size_t i, size_t j, size_t k) {
        COMPUTE_Y2(w, calculate_momentum_rhs_with_forcing_w);
    }

    // Perform the second stage of a single step of an explicit RK3 method for a
    // single point of a single component.
    #define COMPUTE_Y3(component, equation)                                 \
        const Real dt = U.constants.dt;                                     \
        return U.component(i,j,k) + dt * (a31 * equation(U, i, j, k, t_n) + \
               a32 * equation(Y2, i, j, k, time_1));                        \
        
    inline Real compute_Y3_u(const VelocityTensor &U,
                             const VelocityTensor &Y2,
                             Real t_n, Real time_1,
                             size_t i, size_t j, size_t k) {
        COMPUTE_Y3(u, calculate_momentum_rhs_with_forcing_u);
    }

    inline Real compute_Y3_v(const VelocityTensor &U,
                             const VelocityTensor &Y2,
                             Real t_n, Real time_1,
                             size_t i, size_t j, size_t k) {
        COMPUTE_Y3(v, calculate_momentum_rhs_with_forcing_v);
    }

    inline Real compute_Y3_w(const VelocityTensor &U,
                             const VelocityTensor &Y2,
                             Real t_n, Real time_1,
                             size_t i, size_t j, size_t k) {
        COMPUTE_Y3(w, calculate_momentum_rhs_with_forcing_w);
    }

    // Perform the third stage of a single step of an explicit RK3 method for a
    // single point of a single component.
    #define COMPUTE_U_STAR(component, equation)                            \
        const Real dt = U.constants.dt;                                    \
        return U.component(i,j,k) + dt * (b1 * equation(U, i, j, k, t_n) + \
               b3 * equation(Y3, i, j, k, time_2));                        \
        
    inline Real compute_u_star_u(const VelocityTensor &U,
                                 const VelocityTensor &Y3,
                                 Real t_n, Real time_2,
                                 size_t i, size_t j, size_t k) {
        COMPUTE_U_STAR(u, calculate_momentum_rhs_with_forcing_u);
    }

    inline Real compute_u_star_v(const VelocityTensor &U,
                                 const VelocityTensor &Y3,
                                 Real t_n, Real time_2,
                                 size_t i, size_t j, size_t k) {
        COMPUTE_U_STAR(v, calculate_momentum_rhs_with_forcing_v);
    }

    inline Real compute_u_star_w(const VelocityTensor &U,
                                 const VelocityTensor &Y3,
                                 Real t_n, Real time_2,
                                 size_t i, size_t j, size_t k) {
        COMPUTE_U_STAR(w, calculate_momentum_rhs_with_forcing_w);
    }


    void timestep(VelocityTensor &velocity,
                  VelocityTensor &velocity_buffer1,
                  VelocityTensor &velocity_buffer2,
                  Real t_n) {
        const Constants &constants = velocity.constants;
        const Real time_1 = t_n + c2*constants.dt;
        const Real time_2 = t_n + c3*constants.dt;
        const Real final_time = t_n + constants.dt;

        // Stage 1.
        // Apply Dirichlet boundary conditions.
        velocity_buffer1.apply_all_dirichlet_bc(time_1);

        // Compute the solution inside the domain.
        VELOCITY_TENSOR_SET_FOR_ALL_POINTS(
            velocity_buffer1, compute_Y2_u, compute_Y2_v, compute_Y2_w,
            false, velocity, t_n, i, j, k)

        // Stage 2.
        // Apply Dirichlet boundary conditions.
        velocity_buffer2.apply_all_dirichlet_bc(time_2);

        // Compute the solution inside the domain.
        VELOCITY_TENSOR_SET_FOR_ALL_POINTS(
            velocity_buffer2, compute_Y3_u, compute_Y3_v, compute_Y3_w,
            false, velocity, velocity_buffer1, t_n, time_1, i, j, k)

        // Stage 3.
        // Apply Dirichlet boundary conditions.
        velocity_buffer1.apply_all_dirichlet_bc(final_time);

        // Compute the solution inside the domain.
        VELOCITY_TENSOR_SET_FOR_ALL_POINTS(
            velocity_buffer1, compute_u_star_u, compute_u_star_v, compute_u_star_w,
            false, velocity, velocity_buffer2, t_n, time_2, i, j, k)

        // Put the solution in the original tensors.
        velocity.swap_data(velocity_buffer1);
    }

}
