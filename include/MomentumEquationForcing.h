#ifndef MOMENTUM_EQUATION_FORCING_H
#define MOMENTUM_EQUATION_FORCING_H

#include "ManufacturedVelocity.h"
#include "MomentumEquation.h"

namespace mif {

// Add the forcing term to the momentum equation.
inline Real calculate_momentum_rhs_with_forcing_u(
    const VelocityTensor &velocity,                 // Velocity field.
    const size_t i, const size_t j, const size_t k, // Grid point.
    const Real time) {
  return calculate_momentum_rhs_u(velocity, i, j, k) +
         velocity.u.evaluate_function_at_index(time, i, j, k, forcing_x);
}

inline Real calculate_momentum_rhs_with_forcing_v(
    const VelocityTensor &velocity,                 // Velocity field.
    const size_t i, const size_t j, const size_t k, // Grid point.
    const Real time) {
  return calculate_momentum_rhs_v(velocity, i, j, k) +
         velocity.v.evaluate_function_at_index(time, i, j, k, forcing_y);
}

inline Real calculate_momentum_rhs_with_forcing_w(
    const VelocityTensor &velocity,                 // Velocity field.
    const size_t i, const size_t j, const size_t k, // Grid point.
    const Real time) {
  return calculate_momentum_rhs_w(velocity, i, j, k) +
         velocity.w.evaluate_function_at_index(time, i, j, k, forcing_z);
}

} // namespace mif

#endif // MOMENTUM_EQUATION_FORCING_H
