#ifndef MOMENTUM_EQUATION_H
#define MOMENTUM_EQUATION_H

#include "ManufacturedVelocity.h"
#include "VelocityTensor.h"

// Given a specific point in the grid denoted by the indices i, j, k, the
// following functions calculate the right-hand side of the momentum equation
// for the u, v, and w components, respectively. The incoming parameters are the
// scalar fields u, v, and w, which represent the velocity components in the x,
// y, and z directions, respectively.

// We're suggesting the compiler to inline these functions, since they're very
// small and called many times.

namespace mif {

// Compute the diffusion term in the momentum equation for a given velocity
// component. This is mainly used in the momentum equation calculation in order
// to reduce code duplication.
inline Real calculate_diffusion_term(const StaggeredTensor *component,
                                     const size_t i, const size_t j,
                                     const size_t k) {
  const Constants &constants = component->constants;

  const Real result = ((*component)(i + 1, j, k) - 2 * (*component)(i, j, k) +
                       (*component)(i - 1, j, k)) *
                          constants.one_over_dx2_Re +
                      ((*component)(i, j + 1, k) - 2 * (*component)(i, j, k) +
                       (*component)(i, j - 1, k)) *
                          constants.one_over_dy2_Re +
                      ((*component)(i, j, k + 1) - 2 * (*component)(i, j, k) +
                       (*component)(i, j, k - 1)) *
                          constants.one_over_dz2_Re;

  return result;
}

// The momentum equation is composed by two terms: convection and diffusion.
//  -  (u * ∇) u = u ∂u/∂x + v ∂u/∂y + w ∂u/∂z
//  -  1/Re * ∇²u = 1/Re * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
inline Real
calculate_momentum_rhs_u(const VelocityTensor &velocity, // Velocity field.
                         const size_t i, const size_t j, const size_t k) {
  const Constants &constants = velocity.constants;
  const UTensor &u = velocity.u;
  const VTensor &v = velocity.v;
  const WTensor &w = velocity.w;
#ifdef OPT_CPU_CACHE
  const auto &var13 = u(i - 1, j, k);
  const auto &var8 = u(i, j - 1, k);
  const auto &var4 = u(i, j, k - 1);
  const auto &var1 = u(i, j, k);
  const auto &var2 = u(i, j, k + 1);
  const auto &var7 = u(i, j + 1, k);
  const auto &var10 = u(i + 1, j, k);

  const auto &var15 = v(i, j - 1, k);
  const auto &var14 = v(i, j, k);
  const auto &var17 = v(i + 1, j - 1, k);
  const auto &var16 = v(i + 1, j, k);

  const auto &var18 = w(i, j, k - 1);
  const auto &var21 = w(i, j, k);
  const auto &var20 = w(i + 1, j, k - 1);
  const auto &var19 = w(i + 1, j, k);

  const Real convection_term =
      // TERM: u ∂u/∂x
      // We use a first-order finite difference scheme to approximate the
      // convection term, this is sufficient for second-order accuracy in the
      // overall scheme.
      -var1 * (var10 - var13) * constants.one_over_2_dx

      // TERM: v ∂u/∂y
      // We use a second-order central difference scheme to approximate the
      // convection term.
      -
      (var14 + var15 + var16 + var17) * (var7 - var8) * constants.one_over_8_dy

      // TERM: w ∂u/∂z
      -
      (var21 + var18 + var19 + var20) * (var2 - var4) * constants.one_over_8_dz;

  // TERM: 1/Re * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
  // We use a second-order central difference scheme to approximate the second
  // derivatives.
  const Real diffusion_term =
      (var10 - 2 * var1 + var13) * constants.one_over_dx2_Re +
      (var7 - 2 * var1 + var8) * constants.one_over_dy2_Re +
      (var2 - 2 * var1 + var4) * constants.one_over_dz2_Re;

  return convection_term + diffusion_term;
#else
  const Real convection_term =
      // TERM: u ∂u/∂x
      // We use a first-order finite difference scheme to approximate the
      // convection term, this is sufficient for second-order accuracy in the
      // overall scheme.
      -u(i, j, k) * (u(i + 1, j, k) - u(i - 1, j, k)) * constants.one_over_2_dx

      // TERM: v ∂u/∂y
      // We use a second-order central difference scheme to approximate the
      // convection term.
      - (v(i, j + 1, k) + v(i, j, k) + v(i - 1, j + 1, k) + v(i - 1, j, k)) *
            (u(i, j + 1, k) - u(i, j - 1, k)) * constants.one_over_8_dy

      // TERM: w ∂u/∂z
      - (w(i, j, k + 1) + w(i, j, k) + w(i - 1, j, k + 1) + w(i - 1, j, k)) *
            (u(i, j, k + 1) - u(i, j, k - 1)) * constants.one_over_8_dz;

  // TERM: 1/Re * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
  // We use a second-order central difference scheme to approximate the second
  // derivatives.
  const Real diffusion_term =
      calculate_diffusion_term(velocity.components[0], i, j, k);

  return convection_term + diffusion_term;
#endif
}

// Calculate the right-hand side of the momentum equation for the v component.
inline Real
calculate_momentum_rhs_v(const VelocityTensor &velocity, // Velocity field.
                         const size_t i, const size_t j, const size_t k) {
  const Constants &constants = velocity.constants;
  const UTensor &u = velocity.u;
  const VTensor &v = velocity.v;
  const WTensor &w = velocity.w;
#ifdef OPT_CPU_CACHE
  const auto &var2 = u(i - 1, j, k);
  const auto &var4 = u(i - 1, j + 1, k);
  const auto &var1 = u(i, j, k);
  const auto &var3 = u(i, j + 1, k);

  const auto &var6 = v(i - 1, j, k);
  const auto &var8 = v(i, j - 1, k);
  const auto &var15 = v(i, j, k - 1);
  const auto &var7 = v(i, j, k);
  const auto &var14 = v(i, j, k + 1);
  const auto &var9 = v(i, j + 1, k);
  const auto &var5 = v(i + 1, j, k);

  const auto &var11 = w(i, j, k - 1);
  const auto &var10 = w(i, j, k);
  const auto &var13 = w(i, j + 1, k - 1);
  const auto &var12 = w(i, j + 1, k);

  const Real convection_term =
      // TERM: u ∂v/∂x
      -(var1 + var2 + var3 + var4) * (var5 - var6) * constants.one_over_8_dx

      // TERM: v ∂v/∂y
      - var7 * (var8 - var9) * constants.one_over_2_dy

      // TERM: w ∂v/∂z
      - (var10 + var11 + var12 + var13) * (var14 - var15) *
            constants.one_over_8_dz;

  const Real diffusion_term =
      // TERM: 1/Re * (∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²)
      (var5 - 2 * var7 + var6) * constants.one_over_dx2_Re +
      (var9 - 2 * var7 + var8) * constants.one_over_dy2_Re +
      (var14 - 2 * var7 + var15) * constants.one_over_dz2_Re;

  return convection_term + diffusion_term;
#else
  const Real convection_term =
      // TERM: u ∂v/∂x
      -(u(i + 1, j, k) + u(i, j, k) + u(i + 1, j - 1, k) + u(i, j - 1, k)) *
          (v(i + 1, j, k) - v(i - 1, j, k)) * constants.one_over_8_dx

      // TERM: v ∂v/∂y
      - v(i, j, k) * (v(i, j + 1, k) - v(i, j - 1, k)) * constants.one_over_2_dy

      // TERM: w ∂v/∂z
      - (w(i, j, k + 1) + w(i, j, k) + w(i, j - 1, k + 1) + w(i, j - 1, k)) *
            (v(i, j, k + 1) - v(i, j, k - 1)) * constants.one_over_8_dz;

  // TERM: 1/Re * (∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²)
  const Real diffusion_term =
      calculate_diffusion_term(velocity.components[1], i, j, k);

  return convection_term + diffusion_term;
#endif
}

// Calculate the right-hand side of the momentum equation for the w component.
inline Real
calculate_momentum_rhs_w(const VelocityTensor &velocity, // Velocity field.
                         const size_t i, const size_t j, const size_t k) {
  const Constants &constants = velocity.constants;
  const UTensor &u = velocity.u;
  const VTensor &v = velocity.v;
  const WTensor &w = velocity.w;

#ifdef OPT_CPU_CACHE
  const auto &var2 = u(i - 1, j, k);
  const auto &var4 = u(i - 1, j, k + 1);
  const auto &var1 = u(i, j, k);
  const auto &var3 = u(i, j, k + 1);

  const auto &var8 = v(i, j - 1, k);
  const auto &var10 = v(i, j - 1, k + 1);
  const auto &var7 = v(i, j, k);
  const auto &var9 = v(i, j, k + 1);

  const auto &var6 = w(i - 1, j, k);
  const auto &var12 = w(i, j - 1, k);
  const auto &var15 = w(i, j, k - 1);
  const auto &var13 = w(i, j, k);
  const auto &var14 = w(i, j, k + 1);
  const auto &var11 = w(i, j + 1, k);
  const auto &var5 = w(i + 1, j, k);

  const Real convection_term =
      // TERM: u ∂w/∂x
      -(var1 + var2 + var3 + var4) * (var5 - var6) * constants.one_over_8_dx

      // TERM: v ∂w/∂y
      - (var7 + var8 + var9 + var10) * (var11 - var12) * constants.one_over_8_dy

      // TERM: w ∂w/∂z
      - var13 * (var14 - var15) * constants.one_over_2_dz;

  // TERM: 1/Re * (∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
  const Real diffusion_term =
      // TERM: 1/Re * (∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
      (var5 - 2 * var13 + var6) * constants.one_over_dx2_Re +
      (var11 - 2 * var13 + var12) * constants.one_over_dy2_Re +
      (var14 - 2 * var13 + var15) * constants.one_over_dz2_Re;

  return convection_term + diffusion_term;
#else
  const Real convection_term =
      // TERM: u ∂w/∂x
      -(u(i + 1, j, k) + u(i, j, k) + u(i + 1, j, k - 1) + u(i, j, k - 1)) *
          (w(i + 1, j, k) - w(i - 1, j, k)) * constants.one_over_8_dx

      // TERM: v ∂w/∂y
      - (v(i, j + 1, k) + v(i, j, k) + v(i, j + 1, k - 1) + v(i, j, k - 1)) *
            (w(i, j + 1, k) - w(i, j - 1, k)) * constants.one_over_8_dy

      // TERM: w ∂w/∂z
      -
      w(i, j, k) * (w(i, j, k + 1) - w(i, j, k - 1)) * constants.one_over_2_dz;

  // TERM: 1/Re * (∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
  const Real diffusion_term =
      calculate_diffusion_term(velocity.components[2], i, j, k);

  return convection_term + diffusion_term;
#endif
}

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

#endif // MOMENTUM_EQUATION_H
