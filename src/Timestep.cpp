#include "Timestep.h"
#include "MomentumEquation.h"
#include "PressureEquation.h"
#include "PressureGradient.h"
#include "StaggeredTensorMacros.h"

namespace mif {

// Compute a component of Y2 (first step of the method).
#define COMPUTE_COMPONENT_Y2(component) {                                                     \
  const Real dt = velocity.constants.dt;                                                      \
  const Real a1 = 64.0 / 120.0 * dt;                                                          \
  const Real b = a1;                                                                          \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                         \
    const Real rhs =                                                                          \
        calculate_momentum_rhs_##component(velocity, i, j, k);                                \
    const Real p_grad = pressure_gradient_##component(pressure, i, j, k);                     \
    velocity_buffer.component(i, j, k) = velocity.component(i, j, k) + a1 * rhs - b * p_grad; \
    velocity_buffer_2.component(i, j, k) = rhs;                                               \
  )                                                                                           \
}

// Compute a component of Y3 (second step of the method).
#define COMPUTE_COMPONENT_Y3(component) {                                                     \
  const Real dt = velocity.constants.dt;                                                      \
  const Real a1 = -34.0 / 120.0 * dt;                                                         \
  const Real a2 = 50.0 / 120.0 * dt;                                                          \
  const Real b = a1+a2;                                                                       \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity_buffer_2.component, false,                \
    const Real rhs_1 = velocity_buffer_2.component(i, j, k);                                  \
    const Real rhs_2 =                                                                        \
        calculate_momentum_rhs_##component(velocity_buffer, i, j, k);                         \
    const Real rhs_2_scaled = a2 * rhs_2;                                                     \
    const Real p_grad = pressure_gradient_##component(pressure, i, j, k);                     \
    velocity_buffer_2.component(i, j, k) = velocity_buffer.component(i, j, k) +               \
        a1 * rhs_1 + rhs_2_scaled - b * p_grad;                                               \
    velocity.component(i, j, k) = rhs_2_scaled;                                               \
  )                                                                                           \
}

// Compute a component of U* (third and last step of the method).
#define COMPUTE_COMPONENT_U_STAR(component) {                                                 \
  const Real dt = velocity.constants.dt;                                                      \
  const Real a2 = -50.0 / 120.0 * dt;                                                         \
  const Real a3 = 90.0 / 120.0 * dt;                                                          \
  const Real b = a2+a3;                                                                       \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                         \
    const Real rhs_2_scaled = -velocity.component(i, j, k);                                   \
    const Real rhs_3 = calculate_momentum_rhs_##component(velocity_buffer_2, i, j, k);        \
    const Real p_grad = pressure_gradient_##component(pressure, i, j, k);                     \
    velocity.component(i, j, k) = velocity_buffer_2.component(i,j,k) +                        \
        rhs_2_scaled + a3 * rhs_3 - b * p_grad;                                               \
  )                                                                                           \
}

// Compute all components of Y2/Y3/U*.
// "step" should be Y2, Y3 or U_STAR.
#define COMPUTE_STEP(step) {                                       \
  COMPUTE_COMPONENT_##step(u)                                      \
  COMPUTE_COMPONENT_##step(v)                                      \
  COMPUTE_COMPONENT_##step(w)                                      \
}

// Add the pressure gradient adjustment to the velocity.
// This requires MPI communication for the update to the ghost points.
void update_velocity_pressure_gradient(VelocityTensor &velocity, const StaggeredTensor &pressure, Real delta_time) {
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.u, false, velocity.u(i,j,k) -= pressure_gradient_u(pressure, i, j, k) * delta_time;)
  velocity.u.send_mpi_data(0);                                                                                                                 
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.v, false, velocity.v(i,j,k) -= pressure_gradient_v(pressure, i, j, k) * delta_time;)
  velocity.v.send_mpi_data(4);                                                                                                                 
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.w, false, velocity.w(i,j,k) -= pressure_gradient_w(pressure, i, j, k) * delta_time;)
  velocity.w.send_mpi_data(8);                                                                                                                 
  velocity.u.receive_mpi_data(0);                                                                                                              
  velocity.v.receive_mpi_data(4);                                                                                                              
  velocity.w.receive_mpi_data(8);   
}

// Add the pressure difference to the pressure.
void update_pressure(StaggeredTensor &pressure, const StaggeredTensor &pressure_difference) {
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(pressure, true, pressure(i,j,k) += pressure_difference(i,j,k);)
}

// Solve the pressure equation (different function call based on boundary conditions).
#define PRESSURE_EQUATION_false(velocity, dt, new_time, prev_time) {                     \
  solve_pressure_equation_homogeneous_periodic(pressure_buffer, pressure_solver_buffer,  \
                                               velocity, dt);                            \
}

#define PRESSURE_EQUATION_true(velocity, dt, new_time, prev_time) {                                                       \
  solve_pressure_equation_non_homogeneous_neumann(pressure_buffer, pressure_solver_buffer,                                \
                                                  velocity,                                                               \
                                                  exact_pressure_gradient.get_difference_over_time(new_time, prev_time),  \
                                                  dt);                                                                    \
}

// Perform a timestep (different function calls to the pressure solver based on boundary conditions).
#define FULL_TIMESTEP(nhn) {                                                                                              \
  const Constants &constants = velocity.constants;                                                                        \
  const Real dt_1 = 64.0 / 120.0 * constants.dt;                                                                          \
  const Real time_1 = t_n + dt_1;                                                                                         \
  const Real dt_2 = 16.0 / 120.0 * constants.dt;                                                                          \
  const Real time_2 = time_1 + dt_2;                                                                                      \
  const Real dt_3 = 40.0 / 120.0 * constants.dt;                                                                          \
  const Real final_time = t_n + constants.dt;                                                                             \
                                                                                                                          \
  /* Stage 1. */                                                                                                          \
  /* Compute the velocity solution inside the domain. */                                                                  \
  COMPUTE_STEP(Y2)                                                                                                        \
  /* Apply Dirichlet boundary conditions to the velocity. */                                                              \
  velocity_buffer.apply_bc(exact_velocity.set_time(time_1));                                                              \
  /* Solve the pressure equation. */                                                                                      \
  PRESSURE_EQUATION_##nhn(velocity_buffer, dt_1, time_1, t_n)                                                             \
  /* Update the pressure. */                                                                                              \
  update_pressure(pressure, pressure_buffer);                                                                             \
  /* Update the velocity. */                                                                                              \
  update_velocity_pressure_gradient(velocity_buffer, pressure_buffer, dt_1);                                              \
                                                                                                                          \
  /* Stage 2. */                                                                                                          \
  /* Compute the velocity solution inside the domain. */                                                                  \
  COMPUTE_STEP(Y3)                                                                                                        \
  /* Apply Dirichlet boundary conditions to the velocity. */                                                              \
  velocity_buffer_2.apply_bc(exact_velocity.set_time(time_2));                                                            \
  /* Solve the pressure equation. */                                                                                      \
  PRESSURE_EQUATION_##nhn(velocity_buffer_2, dt_2, time_2, time_1)                                                        \
  /* Update the pressure. */                                                                                              \
  update_pressure(pressure, pressure_buffer);                                                                             \
  /* Update the velocity. */                                                                                              \
  update_velocity_pressure_gradient(velocity_buffer_2, pressure_buffer, dt_2);                                            \
                                                                                                                          \
  /* Stage 3. */                                                                                                          \
  /* Compute the velocity solution inside the domain. */                                                                  \
  COMPUTE_STEP(U_STAR)                                                                                                    \
  /* Apply Dirichlet boundary conditions to the velocity. */                                                              \
  velocity.apply_bc(exact_velocity.set_time(final_time));                                                                 \
  /* Solve the pressure equation. */                                                                                      \
  PRESSURE_EQUATION_##nhn(velocity, dt_3, final_time, time_2)                                                             \
  /* Update the pressure. */                                                                                              \
  update_pressure(pressure, pressure_buffer);                                                                             \
  /* Update the velocity. */                                                                                              \
  update_velocity_pressure_gradient(velocity, pressure_buffer, dt_3);                                                     \
}

void timestep_nhn(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
                  VelocityTensor &velocity_buffer_2, const TimeVectorFunction &exact_velocity,
                  const TimeVectorFunction &exact_pressure_gradient, Real t_n,
                  StaggeredTensor &pressure, StaggeredTensor &pressure_buffer, 
                  PressureTensor &pressure_solver_buffer) {
  FULL_TIMESTEP(true)
}

void timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
              VelocityTensor &velocity_buffer_2, const TimeVectorFunction &exact_velocity,
              Real t_n, StaggeredTensor &pressure, StaggeredTensor &pressure_buffer, 
              PressureTensor &pressure_solver_buffer) {
  FULL_TIMESTEP(false)
}

} // namespace mif
