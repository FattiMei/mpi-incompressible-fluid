#ifndef MOMENTUM_EQUATION_H
#define MOMENTUM_EQUATION_H

#include "VelocityTensor.h"

// Given a specific point in the grid denoted by the indices i, j, k, the
// following functions calculate the right-hand side of the momentum equation
// for the u, v, and w components, respectively. The incoming parameters are the
// scalar fields u, v, and w, which represent the velocity components in the x,
// y, and z directions, respectively. The pressure gradient and forcing terms are
// not included.

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
  // For vector u
  const auto& var13_u = u(i - 1, j, k);
  const auto& var11_u = u(i, j - 1, k);
  const auto& var5_u = u(i, j, k - 1);
  const auto& var14_u = u(i, j, k);
  const auto& var23_u = u(i, j, k + 1);
  const auto& var17_u = u(i, j + 1, k);
  const auto& var15_u = u(i + 1, j, k);

  // For vector v
  const auto& var13_v = v(i - 1, j, k);
  const auto& var16_v = v(i - 1, j + 1, k);
  const auto& var14_v = v(i, j, k);
  const auto& var17_v = v(i, j + 1, k);

  // For vector w
  const auto& var13_w = w(i - 1, j, k);
  const auto& var22_w = w(i - 1, j, k + 1);
  const auto& var14_w = w(i, j, k);
  const auto& var23_w = w(i, j, k + 1);

  const Real convection_term =
    // TERM: u ∂u/∂x
    // We use a first-order finite difference scheme to approximate the
    // convection term, this is sufficient for second-order accuracy in the
    // overall scheme.
    -var14_u * (var15_u - var13_u) * constants.one_over_2_dx

    // TERM: v ∂u/∂y
    // We use a second-order central difference scheme to approximate the
    // convection term.
    - (var17_v + var14_v + var16_v + var13_v) *
        (var17_u - var11_u) * constants.one_over_8_dy

    // TERM: w ∂u/∂z
    - (var23_w + var14_w + var22_w + var13_w) *
        (var23_u - var5_u) * constants.one_over_8_dz;

  // TERM: 1/Re * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
  const Real diffusion_term =
      (var15_u - 2 * var14_u + var13_u) * constants.one_over_dx2_Re +
      (var17_u - 2 * var14_u + var11_u) * constants.one_over_dy2_Re +
      (var23_u - 2 * var14_u + var5_u) * constants.one_over_dz2_Re;

  return convection_term + diffusion_term;
#else
  const Real convection_term =
      // TERM: u ∂u/∂x
      -u(i, j, k) * (u(i + 1, j, k) - u(i - 1, j, k)) * constants.one_over_2_dx

      // TERM: v ∂u/∂y
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
  // For vector u
  const auto& var11_u = u(i, j - 1, k);
  const auto& var14_u = u(i, j, k);
  const auto& var12_u = u(i + 1, j - 1, k);
  const auto& var15_u = u(i + 1, j, k);

  // For vector v
  const auto& var13_v = v(i - 1, j, k);
  const auto& var11_v = v(i, j - 1, k);
  const auto& var5_v = v(i, j, k - 1);
  const auto& var14_v = v(i, j, k);
  const auto& var23_v = v(i, j, k + 1);
  const auto& var17_v = v(i, j + 1, k);
  const auto& var15_v = v(i + 1, j, k);

  // For vector w
  const auto& var11_w = w(i, j - 1, k);
  const auto& var20_w = w(i, j - 1, k + 1);
  const auto& var14_w = w(i, j, k);
  const auto& var23_w = w(i, j, k + 1);

  const Real convection_term =
      // TERM: u ∂v/∂x
      -(var15_u + var14_u + var12_u + var11_u) * (var15_v - var13_v) * constants.one_over_8_dx

      // TERM: v ∂v/∂y
      - var14_v * (var17_v - var11_v) * constants.one_over_2_dy

      // TERM: w ∂v/∂z
      - (var23_w + var14_w + var20_w + var11_w) * (var23_v - var5_v) *
            constants.one_over_8_dz;

  const Real diffusion_term =
      // TERM: 1/Re * (∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²)
      (var15_v - 2 * var14_v + var13_v) * constants.one_over_dx2_Re +
      (var17_v - 2 * var14_v + var11_v) * constants.one_over_dy2_Re +
      (var23_v - 2 * var14_v + var5_v) * constants.one_over_dz2_Re;

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
  // For vector u
  const auto& var5_u = u(i, j, k - 1);
  const auto& var14_u = u(i, j, k);
  const auto& var6_u = u(i + 1, j, k - 1);
  const auto& var15_u = u(i + 1, j, k);

  // For vector v
  const auto& var5_v = v(i, j, k - 1);
  const auto& var14_v = v(i, j, k);
  const auto& var8_v = v(i, j + 1, k - 1);
  const auto& var17_v = v(i, j + 1, k);

  // For vector w
  const auto& var13_w = w(i - 1, j, k);
  const auto& var11_w = w(i, j - 1, k);
  const auto& var5_w = w(i, j, k - 1);
  const auto& var14_w = w(i, j, k);
  const auto& var23_w = w(i, j, k + 1);
  const auto& var17_w = w(i, j + 1, k);
  const auto& var15_w = w(i + 1, j, k);

  const Real convection_term =
      // TERM: u ∂w/∂x
      -(var15_u + var14_u + var6_u + var5_u) * (var15_w - var13_w) * constants.one_over_8_dx

      // TERM: v ∂w/∂y
      - (var17_v + var14_v + var8_v + var5_v) * (var17_w - var11_w) * constants.one_over_8_dy

      // TERM: w ∂w/∂z
      - var14_w * (var23_w - var5_w) * constants.one_over_2_dz;

  // TERM: 1/Re * (∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
  const Real diffusion_term =
      // TERM: 1/Re * (∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
      (var15_w - 2 * var14_w + var13_w) * constants.one_over_dx2_Re +
      (var17_w - 2 * var14_w + var11_w) * constants.one_over_dy2_Re +
      (var23_w - 2 * var14_w + var5_w) * constants.one_over_dz2_Re;

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
      - w(i, j, k) * (w(i, j, k + 1) - w(i, j, k - 1)) * constants.one_over_2_dz;

  // TERM: 1/Re * (∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
  const Real diffusion_term =
      calculate_diffusion_term(velocity.components[2], i, j, k);

  return convection_term + diffusion_term;
#endif
}

} // namespace mif

#endif // MOMENTUM_EQUATION_H
