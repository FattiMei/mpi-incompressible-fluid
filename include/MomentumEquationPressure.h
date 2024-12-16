#ifndef MOMENTUM_EQUATION_FORCING_H
#define MOMENTUM_EQUATION_FORCING_H

#include "ManufacturedVelocity.h"
#include "MomentumEquation.h"

namespace mif {

// Compute dp/dx in the staggered point (i,j,k) staggered in the x direction.
inline Real pressure_gradient_x(const StaggeredTensor &pressure,
                                const size_t i, const size_t j, const size_t k) {
    return (pressure(i,j,k) - pressure(i-1,j,k)) * pressure.constants.one_over_dx;
}

inline Real pressure_gradient_y(const StaggeredTensor &pressure,
                                const size_t i, const size_t j, const size_t k) {
    return (pressure(i,j,k) - pressure(i,j-1,k)) * pressure.constants.one_over_dy;
}

inline Real pressure_gradient_z(const StaggeredTensor &pressure,
                                const size_t i, const size_t j, const size_t k) {
    return (pressure(i,j,k) - pressure(i,j,k-1)) * pressure.constants.one_over_dz;
}

// Add the pressure gradient to the momentum equation.
inline Real calculate_momentum_rhs_with_pressure_u(
    const VelocityTensor &velocity,                   // Velocity field.
    const StaggeredTensor &pressure,                  // Pressure field.
    const size_t i, const size_t j, const size_t k) { // Grid point.
  return calculate_momentum_rhs_u(velocity, i, j, k) - pressure_gradient_x(pressure, i, j, k);
}

inline Real calculate_momentum_rhs_with_pressure_v(
    const VelocityTensor &velocity,                   // Velocity field.
    const StaggeredTensor &pressure,                  // Pressure field.
    const size_t i, const size_t j, const size_t k) { // Grid point.
  return calculate_momentum_rhs_v(velocity, i, j, k) - pressure_gradient_y(pressure, i, j, k);
}

inline Real calculate_momentum_rhs_with_pressure_w(
    const VelocityTensor &velocity,                   // Velocity field.
    const StaggeredTensor &pressure,                  // Pressure field.
    const size_t i, const size_t j, const size_t k) { // Grid point.
  return calculate_momentum_rhs_w(velocity, i, j, k) - pressure_gradient_z(pressure, i, j, k);
}

} // namespace mif

#endif // MOMENTUM_EQUATION_FORCING_H
