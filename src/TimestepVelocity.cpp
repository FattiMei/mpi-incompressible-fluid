//
// Created by giorgio on 10/10/2024.
//

#include "TimestepVelocity.h"
#include "MomentumEquationForcing.h"
#include "StaggeredTensorMacros.h"

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
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                       \
    const Real dt = velocity.constants.dt;                                                  \
    const Real rhs =                                                                        \
        calculate_momentum_rhs_with_forcing_##component(velocity, i, j, k, t_n);            \
    rhs_buffer.component(i,j,k) = rhs;                                                      \
    velocity_buffer.component(i, j, k) = velocity.component(i, j, k) + dt * a21 * rhs;      \
  )                                                                                         \
}

// Compute a component of Y3 (second step of the method).
#define COMPUTE_COMPONENT_Y3(component) {                                                   \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                       \
    const Real dt = velocity.constants.dt;                                                  \
    const Real rhs = rhs_buffer.component(i,j,k);                                           \
    rhs_buffer.component(i,j,k) = velocity.component(i, j, k) + dt * (b1 * rhs);            \
    velocity.component(i, j, k) = velocity.component(i, j, k) +                             \
              dt * (a31 * rhs + a32 * calculate_momentum_rhs_with_forcing_##component(      \
                                            velocity_buffer, i, j, k, time_1));             \
  )                                                                                         \
}

// Compute a component of U* (third and last step of the method).
#define COMPUTE_COMPONENT_U_STAR(component) {                                               \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                       \
    const Real dt = velocity.constants.dt;                                                  \
    const Real rhs = rhs_buffer.component(i,j,k);                                           \
    velocity_buffer.component(i, j, k) = rhs + dt * (b3 *                                   \
              calculate_momentum_rhs_with_forcing_##component(velocity, i, j, k, time_2));  \
  )                                                                                         \
}

// Compute all components of Y2/Y3/U*.
// "step" should be Y2, Y3 or U_STAR.
#define COMPUTE_STEP(step) {                                       \
  COMPUTE_COMPONENT_##step(u)                                      \
  COMPUTE_COMPONENT_##step(v)                                      \
  COMPUTE_COMPONENT_##step(w)                                      \
}

void timestep_velocity(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
                       VelocityTensor &rhs_buffer, const TimeVectorFunction &exact_velocity, Real t_n) {
  const Constants &constants = velocity.constants;
  const Real time_1 = t_n + c2 * constants.dt;
  const Real time_2 = t_n + c3 * constants.dt;
  const Real final_time = t_n + constants.dt;

  // Stage 1.
  // Compute the solution inside the domain.
  COMPUTE_STEP(Y2)

  // Apply Dirichlet boundary conditions.
  velocity_buffer.apply_bc(exact_velocity.set_time(time_1));

  // Stage 2.
  // Compute the solution inside the domain.
  COMPUTE_STEP(Y3)

  // Apply Dirichlet boundary conditions.
  velocity.apply_bc(exact_velocity.set_time(time_2));

  // Stage 3.
  // Compute the solution inside the domain.
  COMPUTE_STEP(U_STAR)

  // Apply Dirichlet boundary conditions.
  velocity_buffer.apply_bc(exact_velocity.set_time(final_time));

  // Put the solution in the original tensors.
  velocity.swap_data(velocity_buffer);
}

} // namespace mif
