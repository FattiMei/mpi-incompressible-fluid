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

    // Perform the first stage of a single step of an explicit RK3 method for a single point of a single component.
    VelocityTensor::IndexVectorFunction
    compute_Y2(const VelocityTensor &U, 
               const VectorFunction &forcing_term_t_n) {
        const Real dt = U.constants.dt;
        VelocityTensor::IndexVectorFunction rhs = calculate_momentum_rhs_with_forcing(U, forcing_term_t_n);
        VelocityTensor::IndexVectorFunction identity = VelocityTensor::IndexVectorFunction::identity(U);
        return identity + rhs * dt * a21;
    }

    // Perform the second stage of a single step of an explicit RK3 method for a single point of a single component.
    VelocityTensor::IndexVectorFunction
    compute_Y3(const VelocityTensor &U, 
               const VelocityTensor &Y2, 
               const VectorFunction &forcing_term_t_n,
               const VectorFunction &forcing_term_time_1) {
        const Real dt = U.constants.dt;
        VelocityTensor::IndexVectorFunction rhs_1 = calculate_momentum_rhs_with_forcing(U, forcing_term_t_n);
        VelocityTensor::IndexVectorFunction rhs_2 = calculate_momentum_rhs_with_forcing(Y2, forcing_term_time_1);
        VelocityTensor::IndexVectorFunction identity = VelocityTensor::IndexVectorFunction::identity(U);
        return identity + rhs_1 * dt * a31 + rhs_2 * dt * a32;
    }

    // Perform the third stage of a single step of an explicit RK3 method for a single point of a single component.
    VelocityTensor::IndexVectorFunction
    compute_new_u(const VelocityTensor &U, 
                  const VelocityTensor &Y3, 
                  const VectorFunction &forcing_term_t_n,
                  const VectorFunction &forcing_term_time_2) {
        const Real dt = U.constants.dt;
        VelocityTensor::IndexVectorFunction rhs_1 = calculate_momentum_rhs_with_forcing(U, forcing_term_t_n);
        VelocityTensor::IndexVectorFunction rhs_2 = calculate_momentum_rhs_with_forcing(Y3, forcing_term_time_2);
        VelocityTensor::IndexVectorFunction identity = VelocityTensor::IndexVectorFunction::identity(U);
        return identity + rhs_1 * dt * b1 + rhs_2 * dt * b3;
    }

    void timestep(VelocityTensor &velocity,
                  VelocityTensor &velocity_buffer1,
                  VelocityTensor &velocity_buffer2,
                  const TimeVectorFunction &exact_velocity,
                  const TimeVectorFunction &forcing_term,
                  Real t_n) {
        const Constants &constants = velocity.constants;
        const Real time_1 = t_n + c2*constants.dt;
        const Real time_2 = t_n + c3*constants.dt;
        const Real final_time = t_n + constants.dt;
        VectorFunction forcing_term_t_n = forcing_term.set_time(t_n);
        VectorFunction forcing_term_time_1 = forcing_term.set_time(time_1);
        VectorFunction forcing_term_time_2 = forcing_term.set_time(time_2);

        // Stage 1.
        // Apply Dirichlet boundary conditions.
        velocity_buffer1.apply_all_dirichlet_bc(exact_velocity.set_time(time_1));

        // Compute the solution inside the domain.
        VelocityTensor::IndexVectorFunction compute_Y2_func = compute_Y2(velocity, forcing_term_t_n);
        velocity_buffer1.set(compute_Y2_func, false);

        // Stage 2.
        // Apply Dirichlet boundary conditions.
        velocity_buffer2.apply_all_dirichlet_bc(exact_velocity.set_time(time_2));

        // Compute the solution inside the domain.
        VelocityTensor::IndexVectorFunction compute_Y3_func = compute_Y3(velocity, velocity_buffer1, forcing_term_t_n, forcing_term_time_1);
        velocity_buffer2.set(compute_Y3_func, false);

        // Stage 3.
        // Apply Dirichlet boundary conditions.
        velocity_buffer1.apply_all_dirichlet_bc(exact_velocity.set_time(final_time));

        // Compute the solution inside the domain.
        VelocityTensor::IndexVectorFunction compute_new_u_func = compute_new_u(velocity, velocity_buffer2, forcing_term_t_n, forcing_term_time_2);
        velocity_buffer1.set(compute_new_u_func, false);

        // Put the solution in the original tensors.
        velocity.swap_data(velocity_buffer1);
    }

}
