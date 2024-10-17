#ifndef MOMENTUM_EQUATION_H
#define MOMENTUM_EQUATION_H

#include "Constants.h"
#include "VelocityTensor.h"

// Given a specific point in the grid denoted by the indices i, j, k, the following functions calculate the
// right-hand side of the momentum equation for the u, v, and w components, respectively. The incoming
// parameters are the scalar fields u, v, and w, which represent the velocity components in the x, y, and z
// directions, respectively.

// We're suggesting the compiler to inline these functions, since they're very small and called many times.
// TODO: Assess wether or not using the () operator of the Tensor class has an impact on performance with
// respect to accessing the data directly.

namespace mif
{

    // The momentum equation is composed by two terms: convection and diffusion.
    //  -  (u * ∇) u = u ∂u/∂x + v ∂u/∂y + w ∂u/∂z
    //  -  1/Re * ∇²u = 1/Re * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    // TODO: I'm not sure if these are the actual names of the terms. If you agree, feel free to remove this comment.
    // In each function, the convection term is calculated first, followed by the diffusion term. Their results are
    // then algebraically summed up and returned.

    inline Real calculate_momentum_rhs_u(const VelocityTensor velocity, // Velocity field.
                                         const size_t i, const size_t j, const size_t k) {  // Grid point.
        const Constants &constants = velocity.constants;
        const UTensor u = velocity.u;
        const VTensor v = velocity.v;
        const WTensor w = velocity.w;

        const Real convection_term =
                // TERM: u ∂u/∂x
                // We use a first-order finite difference scheme to approximate the convection term,
                // this is sufficient for second-order accuracy in the overall scheme.
                - u(i, j, k) *
                (u(i+1, j, k) - u(i-1, j, k)) * constants.one_over_2_dx

                // TERM: v ∂u/∂y
                // We use a second-order central difference scheme to approximate the convection term.
                -(v(i, j, k) + v(i, j-1, k) + v(i+1, j, k) + v(i+1, j-1, k)) *
                (u(i, j+1, k) - u(i, j-1, k)) * constants.one_over_8_dy

                // TERM: w ∂u/∂z
                -(w(i, j, k) + w(i, j, k-1) + w(i+1, j, k) + w(i+1, j, k-1)) *
                (u(i, j, k+1) - u(i, j, k-1)) * constants.one_over_8_dz;

        const Real diffusion_term =
            // TERM: 1/Re * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
            // We use a second-order central difference scheme to approximate the second derivatives.
            (u(i+1, j, k) - 2 * u(i, j, k) + u(i-1, j, k)) * constants.one_over_dx2_Re +
            (u(i, j+1, k) - 2 * u(i, j, k) + u(i, j-1, k)) * constants.one_over_dy2_Re +
            (u(i, j, k+1) - 2 * u(i, j, k) + u(i, j, k-1)) * constants.one_over_dz2_Re;

        return convection_term + diffusion_term;
    }


    // Calculate the right-hand side of the momentum equation for the v component.
    inline Real calculate_momentum_rhs_v(const VelocityTensor velocity, // Velocity field.
                                         const size_t i, const size_t j, const size_t k) {  // Grid point.
        const Constants &constants = velocity.constants;
        const UTensor u = velocity.u;
        const VTensor v = velocity.v;
        const WTensor w = velocity.w;

        const Real convection_term =
                // TERM: u ∂v/∂x
                -(u(i, j, k) + u(i-1, j, k) + u(i, j+1, k) + u(i-1, j+1, k)) *
                (v(i+1, j, k) - v(i-1, j, k)) * constants.one_over_8_dx

                // TERM: v ∂v/∂y
                - v(i, j, k) *
                (v(i, j+1, k) - v(i, j-1, k)) * constants.one_over_2_dy

                // TERM: w ∂v/∂z
                -(w(i, j, k) + w(i, j, k-1) + w(i, j+1, k) + w(i, j+1, k-1)) *
                (v(i, j, k+1) - v(i, j, k-1)) * constants.one_over_8_dz;

        const Real diffusion_term =
                // TERM: 1/Re * (∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²)
                (v(i+1, j, k) - 2 * v(i, j, k) + v(i-1, j, k)) * constants.one_over_dx2_Re +
                (v(i, j+1, k) - 2 * v(i, j, k) + v(i, j-1, k)) * constants.one_over_dy2_Re +
                (v(i, j, k+1) - 2 * v(i, j, k) + v(i, j, k-1)) * constants.one_over_dz2_Re;

        return convection_term + diffusion_term;
    }


    // Calculate the right-hand side of the momentum equation for the w component.
    inline Real calculate_momentum_rhs_w(const VelocityTensor velocity, // Velocity field.
                                         const size_t i, const size_t j, const size_t k) {  // Grid point.
        const Constants &constants = velocity.constants;
        const UTensor u = velocity.u;
        const VTensor v = velocity.v;
        const WTensor w = velocity.w;

        const Real convection_term =
                // TERM: u ∂w/∂x
                -(u(i, j, k) + u(i-1, j, k) + u(i, j, k+1) + u(i-1, j, k+1)) *
                (w(i+1, j, k) - w(i-1, j, k)) * constants.one_over_8_dx

                // TERM: v ∂w/∂y
                -(v(i, j, k) + v(i, j-1, k) + v(i, j, k+1) + v(i, j-1, k+1)) *
                (w(i, j+1, k) - w(i, j-1, k)) * constants.one_over_8_dy

                // TERM: w ∂w/∂z
                - w(i, j, k) * (w(i, j, k+1) - w(i, j, k-1)) * constants.one_over_2_dz;

        const Real diffusion_term =
            // TERM: 1/Re * (∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
            (w(i+1, j, k) - 2 * w(i, j, k) + w(i-1, j, k)) * constants.one_over_dx2_Re +
            (w(i, j+1, k) - 2 * w(i, j, k) + w(i, j-1, k)) * constants.one_over_dy2_Re +
            (w(i, j, k+1) - 2 * w(i, j, k) + w(i, j, k-1)) * constants.one_over_dz2_Re;

        return convection_term + diffusion_term;
    }


    VelocityTensor::IndexVectorFunction
    calculate_momentum_rhs_with_forcing(const VelocityTensor &velocity,        // Velocity field.
                                        const VectorFunction &forcing_term) {
        const std::function<Real(size_t, size_t, size_t)> f_u =
            [&velocity, &forcing_term](const size_t i, const size_t j, const size_t k) {
                return calculate_momentum_rhs_u(velocity, i, j, k) + velocity.u.evaluate_function_at_index(i, j, k, forcing_term.f_u);
            };
        const std::function<Real(size_t, size_t, size_t)> f_v =
            [&velocity, &forcing_term](const size_t i, const size_t j, const size_t k) {
                return calculate_momentum_rhs_v(velocity, i, j, k) + velocity.v.evaluate_function_at_index(i, j, k, forcing_term.f_v);
            };
        const std::function<Real(size_t, size_t, size_t)> f_w =
            [&velocity, &forcing_term](const size_t i, const size_t j, const size_t k) {
                return calculate_momentum_rhs_w(velocity, i, j, k) + velocity.w.evaluate_function_at_index(i, j, k, forcing_term.f_w);
            };
        return VelocityTensor::IndexVectorFunction(f_u, f_v, f_w);
    }

} // mif

#endif // MOMENTUM_EQUATION_H