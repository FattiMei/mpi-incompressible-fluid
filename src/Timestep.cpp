//
// Created by giorgio on 10/10/2024.
//

#include "Timestep.h"
#include "MomentumEquation.h"
#include "VelocityTensorMacros.h"
#include <cmath>

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
                                                 \
    const Real rhs =                                                                        \
        calculate_momentum_rhs_with_forcing_##component(velocity, i, j, k, t_n);            \
    rhs_buffer.component(i,j,k) = rhs;                                                      \
    velocity_buffer.component(i, j, k) = velocity.component(i, j, k) + dt * a21 * rhs;     \
  )                                                                                         \
}

// Compute a component of Y3 (second step of the method).
#define COMPUTE_COMPONENT_Y3(component) {                                                   \
  VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.component, false,                        \
                                                   \
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
                                                     \
    const Real rhs = rhs_buffer.component(i,j,k);                                           \
    velocity_buffer.component(i, j, k) = rhs + dt * (b3 *                                  \
              calculate_momentum_rhs_with_forcing_##component(velocity, i, j, k, time_2));  \
  )                                                                                         \
}

// Compute all components of Y2/Y3/U*.
// "step" should be Y2, Y3 or U_STAR.
#define COMPUTE_STEP(step) {  \
  COMPUTE_COMPONENT_##step(u) \
  COMPUTE_COMPONENT_##step(v) \
  COMPUTE_COMPONENT_##step(w) \
}

  Real timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
                VelocityTensor &rhs_buffer, Real t_n,Real target_cfl,Real last_dt) {
    const Constants &constants = velocity.constants;

    auto max_velocity = -1.0;
    for (size_t i = 0; i < constants.Nx - 2; i++) {
      for (size_t j = 0; j < constants.Ny - 2; j++) {
        for (size_t k = 0; k < constants.Nz - 2; k++) {
          max_velocity = std::abs(std::max(max_velocity,
                                           std::max(std::abs(velocity.u(i, j, k)),
                                                    std::max(std::abs(velocity.v(i, j, k)),
                                                             std::abs(velocity.w(i, j, k))))));
        }
      }
    }
    /*  std::cout << "CFL condition: " << max_velocity * constants.dt / constants.dx << std::endl;
      std::cout << "Target CFL condition: " << 0.5 << std::endl;*/

    Real target_dt = constants.dx * target_cfl / max_velocity;
    Real dt = std::clamp(target_dt, 0.5 * last_dt, 2.0 * last_dt);


    const Real time_1 = t_n + c2 * dt;
    const Real time_2 = t_n + c3 * dt;
    const Real final_time = t_n + dt;


    // Stage 1.
  // Apply Dirichlet boundary conditions.
  velocity_buffer.apply_all_dirichlet_bc(time_1);

  // Compute the solution inside the domain.
    COMPUTE_STEP(Y2)

    // Stage 2.
    // Apply Dirichlet boundary conditions.
    velocity.apply_all_dirichlet_bc(time_2);

  // Compute the solution inside the domain.
  COMPUTE_STEP(Y3)

  // Stage 3. u_n
  // Apply Dirichlet boundary conmy rk do not converge n time, t dfiverge the more timestep putditions.
    velocity_buffer.apply_all_dirichlet_bc(final_time);

  // Compute the solution inside the domain.
  COMPUTE_STEP(U_STAR)

  // Put the solution in the original tensors.
  velocity.swap_data(velocity_buffer);
  return dt;
}

} // namespace mif
