//
// Created by giorgio on 10/10/2024.
//

#include "Timestep.h"
#include "MomentumEquation.h"
#include "VelocityTensorMacros.h"

namespace mif {

constexpr Real c2 = (8.0 / 15.0);
constexpr Real a21 = (8.0 / 15.0);
constexpr Real c3 = (2.0 / 3.0);
constexpr Real a31 = (1.0 / 4.0);
constexpr Real a32 = (5.0 / 12.0);
constexpr Real b1 = (1.0 / 4.0);
constexpr Real b3 = (3.0 / 4.0);

// Compute a component of Y2 (first step of the method).
#define COMPUTE_COMPONENT_Y2(component) {                                                   \
  VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                        \
    const Real dt = velocity.constants.dt;                                                  \
    const Real rhs =                                                                        \
        calculate_momentum_rhs_with_forcing_##component(velocity, i, j, k, t_n);            \
    rhs_buffer.component(i,j,k) = rhs;                                                      \
    velocity_buffer.component(i, j, k) = velocity.component(i, j, k) + dt * a21 * rhs;     \
  )                                                                                         \
}

// Compute a component of Y3 (second step of the method).
#define COMPUTE_COMPONENT_Y3(component) {                                                   \
  VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                        \
    const Real dt = velocity.constants.dt;                                                  \
    const Real rhs = rhs_buffer.component(i,j,k);                                           \
    rhs_buffer.component(i,j,k) = velocity.component(i, j, k) + dt * (b1 * rhs);            \
    velocity.component(i, j, k) = velocity.component(i, j, k) +                             \
              dt * (a31 * rhs + a32 * calculate_momentum_rhs_with_forcing_##component(      \
                                            velocity_buffer, i, j, k, time_1));            \
  )                                                                                         \
}

// Compute a component of U* (third and last step of the method).
#define COMPUTE_COMPONENT_U_STAR(component) {                                               \
  VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                        \
    const Real dt = velocity.constants.dt;                                                  \
    const Real rhs = rhs_buffer.component(i,j,k);                                           \
    velocity_buffer.component(i, j, k) = rhs + dt * (b3 *                                  \
              calculate_momentum_rhs_with_forcing_##component(velocity, i, j, k, time_2));  \
  )                                                                                         \
}

// Compute all components of Y2/Y3/U*.
// "step" should be Y2, Y3 or U_STAR.

    inline void
    Compute_Y2(const VelocityTensor &velocity, VelocityTensor &velocity_buffer, VelocityTensor &rhs_buffer,
               double t_n) noexcept {
        {
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity.u.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0];
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma omp simd
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const double dt = velocity.constants.dt;
                            double rhs = calculate_momentum_rhs_with_forcing_u(velocity, i, j, k, t_n);
                            rhs_buffer.u(i, j, k) = rhs;
                            velocity_buffer.u(i, j, k) = velocity.u(i, j, k) + dt * a21 * rhs;
                            rhs = calculate_momentum_rhs_with_forcing_v(velocity, i, j, k, t_n);
                            rhs_buffer.v(i, j, k) = rhs;
                            velocity_buffer.v(i, j, k) = velocity.v(i, j, k) + dt * a21 * rhs;
                            rhs = calculate_momentum_rhs_with_forcing_w(velocity, i, j, k, t_n);
                            rhs_buffer.w(i, j, k) = rhs;
                            velocity_buffer.w(i, j, k) = velocity.w(i, j, k) + dt * a21 * rhs;
                        }
                    }
                }
            }
        }

    }

    inline void
    Compute_Y3(VelocityTensor &velocity, const VelocityTensor &velocity_buffer, VelocityTensor &rhs_buffer,
               const double time_1) noexcept {
        {
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity.u.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0];
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma omp simd
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const double dt = velocity.constants.dt;
                            double rhs = rhs_buffer.u(i, j, k);
                            rhs_buffer.u(i, j, k) = velocity.u(i, j, k) + dt * (b1 * rhs);
                            velocity.u(i, j, k) = velocity.u(i, j, k) + dt * (a31 * rhs + a32 *
                                                                                          calculate_momentum_rhs_with_forcing_u(
                                                                                                  velocity_buffer,
                                                                                                  i, j, k, time_1));

                            rhs = rhs_buffer.v(i, j, k);
                            rhs_buffer.v(i, j, k) = velocity.v(i, j, k) + dt * (b1 * rhs);
                            velocity.v(i, j, k) = velocity.v(i, j, k) + dt * (a31 * rhs + a32 *
                                                                                          calculate_momentum_rhs_with_forcing_v(
                                                                                                  velocity_buffer,
                                                                                                  i, j, k, time_1));
                            rhs = rhs_buffer.w(i, j, k);
                            rhs_buffer.w(i, j, k) = velocity.w(i, j, k) + dt * (b1 * rhs);
                            velocity.w(i, j, k) = velocity.w(i, j, k) + dt * (a31 * rhs + a32 *
                                                                                          calculate_momentum_rhs_with_forcing_w(
                                                                                                  velocity_buffer,
                                                                                                  i, j, k, time_1));
                        }
                    }
                }
            }
        }

    }

    void
    Compute_Ustar(const VelocityTensor &velocity, VelocityTensor &velocity_buffer, const VelocityTensor &rhs_buffer,
                  const double time_2) noexcept {
        {
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity.u.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0];
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma omp simd
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const double dt = velocity.constants.dt;
                            double rhs = rhs_buffer.u(i, j, k);
                            velocity_buffer.u(i, j, k) = rhs + dt * (b3 *
                                                                     calculate_momentum_rhs_with_forcing_u(velocity,
                                                                                                           i, j, k,
                                                                                                           time_2));
                            rhs = rhs_buffer.v(i, j, k);
                            velocity_buffer.v(i, j, k) = rhs + dt * (b3 *
                                                                     calculate_momentum_rhs_with_forcing_v(velocity,
                                                                                                           i, j, k,
                                                                                                           time_2));
                            rhs = rhs_buffer.w(i, j, k);
                            velocity_buffer.w(i, j, k) = rhs + dt * (b3 *
                                                                     calculate_momentum_rhs_with_forcing_w(velocity,
                                                                                                           i, j, k,
                                                                                                           time_2));
                        }
                    }
                }
            }
        }

    }

    void timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
                  VelocityTensor &rhs_buffer, Real t_n) {
  const Constants &constants = velocity.constants;
  const Real time_1 = t_n + c2 * constants.dt;
  const Real time_2 = t_n + c3 * constants.dt;
  const Real final_time = t_n + constants.dt;

  // Stage 1.
  // Apply Dirichlet boundary conditions.


  // Compute the solution inside the domain.
        Compute_Y2(velocity, velocity_buffer, rhs_buffer, t_n);
        velocity_buffer.apply_all_dirichlet_bc(time_1);
  // Stage 2.
  // Apply Dirichlet boundary conditions.

  // Compute the solution inside the domain.
        Compute_Y3(velocity, velocity_buffer, rhs_buffer, time_1);

        velocity.apply_all_dirichlet_bc(time_2);


        // Stage 3. u_n
  // Apply Dirichlet boundary conditions.

  // Compute the solution inside the domain.
        Compute_Ustar(velocity, velocity_buffer, rhs_buffer, time_2);
        velocity_buffer.apply_all_dirichlet_bc(final_time);

  // Put the solution in the original tensors.
  velocity.swap_data(velocity_buffer);
}

} // namespace mif
