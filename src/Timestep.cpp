#include "Timestep.h"
#include "MomentumEquation.h"
#include "PressureEquation.h"
#include "PressureGradient.h"
#include "StaggeredTensorMacros.h"

namespace mif {

constexpr Real c2 = (8.0 / 15.0);
constexpr Real a21 = (8.0 / 15.0);
constexpr Real c3 = (2.0 / 3.0);
constexpr Real a31 = (1.0 / 4.0);
constexpr Real a32 = (5.0 / 12.0);
constexpr Real b1 = (1.0 / 4.0);
constexpr Real b3 = (3.0 / 4.0);

constexpr Real d1 = c2;
constexpr Real d2 = c3 - c2;
constexpr Real d3 = 1 - c3;

// Compute a component of Y2 (first step of the method).
#define COMPUTE_COMPONENT_Y2(component, tag) {                                                \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                         \
    const Real dt = velocity.constants.dt;                                                    \
    const Real rhs =                                                                          \
        calculate_momentum_rhs_##component(velocity, i, j, k);                                \
    rhs_buffer.component(i,j,k) = rhs;                                                        \
    velocity_buffer.component(i, j, k) = velocity.component(i, j, k) + dt * a21 * rhs -       \
        dt_1 * pressure_gradient_##component(pressure, i, j, k);                              \
  )                                                                                           \
  velocity_buffer.component.send_mpi_data(tag);                                               \
}

// Compute a component of Y3 (second step of the method).
#define COMPUTE_COMPONENT_Y3(component, tag) {                                                \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                         \
    const Real dt = velocity.constants.dt;                                                    \
    const Real rhs = rhs_buffer.component(i,j,k);                                             \
    rhs_buffer.component(i,j,k) = velocity.component(i, j, k) + dt * (b1 * rhs);              \
    velocity.component(i, j, k) = velocity.component(i, j, k) +                               \
        dt * (a31 * rhs + a32 * calculate_momentum_rhs_##component(                           \
        velocity_buffer, i, j, k)) - dt_2 *                                                   \
        pressure_gradient_##component(pressure, i, j, k);                                     \
  )                                                                                           \
  velocity.component.send_mpi_data(tag);                                                      \
}

// Compute a component of U* (third and last step of the method).
#define COMPUTE_COMPONENT_U_STAR(component, tag) {                                            \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                         \
    const Real dt = velocity.constants.dt;                                                    \
    const Real rhs = rhs_buffer.component(i,j,k);                                             \
    velocity_buffer.component(i, j, k) = rhs + dt * (b3 *                                     \
        calculate_momentum_rhs_##component(velocity, i, j, k)) -                              \
        dt_3 * pressure_gradient_##component(pressure, i, j, k);                              \
  )                                                                                           \
  velocity_buffer.component.send_mpi_data(tag);                                               \
}

// Compute all components of Y2/Y3/U*.
// "step" should be Y2, Y3 or U_STAR.
#define COMPUTE_STEP(step) {                                          \
  COMPUTE_COMPONENT_##step(u, 0)                                      \
  COMPUTE_COMPONENT_##step(v, 4)                                      \
  COMPUTE_COMPONENT_##step(w, 8)                                      \
}

// Add the pressure gradient adjustment to the velocity.
#define UPDATE_VELOCITY(velocity, delta_pressure, delta_time) {                                                                                \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.u, false, velocity.u(i,j,k) -= pressure_gradient_u(delta_pressure, i, j, k) * delta_time;) \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.v, false, velocity.v(i,j,k) -= pressure_gradient_v(delta_pressure, i, j, k) * delta_time;) \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.w, false, velocity.w(i,j,k) -= pressure_gradient_w(delta_pressure, i, j, k) * delta_time;) \
}

#define UPDATE_PRESSURE() {                                                                            \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(pressure, true, pressure(i,j,k) += pressure_buffer(i,j,k);) \
}

void timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
              VelocityTensor &rhs_buffer, const TimeVectorFunction &exact_velocity,
              const TimeVectorFunction &exact_pressure_gradient, Real t_n,
              StaggeredTensor &pressure, StaggeredTensor &pressure_buffer, 
              PressureSolverStructures &structures) {
  const Constants &constants = velocity.constants;
  const Real dt_1 = d1 * constants.dt;
  const Real time_1 = t_n + dt_1;
  const Real dt_2 = d2 * constants.dt;
  const Real time_2 = time_1 + dt_2;
  const Real dt_3 = d3 * constants.dt;
  const Real final_time = t_n + constants.dt;

  // Stage 1.
  // Compute the velocity solution inside the domain.
  COMPUTE_STEP(Y2)
  // Apply Dirichlet boundary conditions to the velocity.
  velocity_buffer.apply_all_dirichlet_bc(exact_velocity.set_time(time_1));
  // Solve the pressure equation.
  solve_pressure_equation_non_homogeneous_neumann(pressure_buffer, velocity_buffer, exact_pressure_gradient.get_difference_over_time(time_1, t_n), structures, dt_1);
  // Update the pressure.
  UPDATE_PRESSURE()
  // Update the velocity.
  UPDATE_VELOCITY(velocity_buffer, pressure_buffer, dt_1)

  // Stage 2.
  // Compute the velocity solution inside the domain.
  COMPUTE_STEP(Y3)
  // Apply Dirichlet boundary conditions  to the velocity.
  velocity.apply_all_dirichlet_bc(exact_velocity.set_time(time_2));
  // Solve the pressure equation.
  solve_pressure_equation_non_homogeneous_neumann(pressure_buffer, velocity, exact_pressure_gradient.get_difference_over_time(time_2, time_1), structures, dt_2);
  // Update the pressure.
  UPDATE_PRESSURE()
  // Update the velocity.
  UPDATE_VELOCITY(velocity, pressure_buffer, dt_2)

  // Stage 3.
  // Compute the velocity solution inside the domain.
  COMPUTE_STEP(U_STAR)
  // Apply Dirichlet boundary conditions  to the velocity.
  velocity_buffer.apply_all_dirichlet_bc(exact_velocity.set_time(final_time));
  // Solve the pressure equation.
  solve_pressure_equation_non_homogeneous_neumann(pressure_buffer, velocity_buffer, exact_pressure_gradient.get_difference_over_time(final_time, time_2), structures, dt_3);
  // Update the pressure.
  UPDATE_PRESSURE()
  // Update the velocity.
  UPDATE_VELOCITY(velocity_buffer, pressure_buffer, dt_3)

  // Put the velocity solution in the original tensors.
  velocity.swap_data(velocity_buffer);
}

} // namespace mif
