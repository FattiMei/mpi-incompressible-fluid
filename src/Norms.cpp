#include "Norms.h"
#include <cmath>

//TODO: using a temporary workaround for approximate norms, but norms need to be computed precisely.

namespace mif {

    Real L2Norm(const VelocityTensor &velocity,
                const VelocityTensor &exact_velocity) {
        Real integral = 0.0;
        const Constants &c = velocity.constants;

        // Iterate over the entire tensor space.
        for (std::size_t i = 0; i < c.Nx; ++i) {
            const Real wxi = (i == 0 || i == c.Nx - 1) ? 0.5 : 1.0;
            for (std::size_t j = 0; j < c.Ny; ++j) {
                const Real wyj = (j == 0 || j == c.Ny - 1) ? 0.5 : 1.0;
                for (std::size_t k = 0; k < c.Nz; ++k) {
                    // Calculate weight for current grid point.
                    const Real wzk = (k == 0 || k == c.Nz - 1) ? 0.5 : 1.0;

                    const Real weight = wxi * wyj * wzk;

                    // Compute differences.
                    const Real diff_u = (i < c.Nx-1) ? velocity.u(i, j, k) - exact_velocity.u(i, j, k) : 0.0;
                    const Real diff_v = (j < c.Ny-1) ? velocity.v(i, j, k) - exact_velocity.v(i, j, k) : 0.0;
                    const Real diff_w = (k < c.Nz-1) ? velocity.w(i, j, k) - exact_velocity.w(i, j, k) : 0.0;

                    // Accumulate squared differences with weights.
                    integral += weight * (diff_u * diff_u + diff_v * diff_v + diff_w * diff_w);
                }
            }
        }

        // Multiply by volume element and return the square root.
        return std::sqrt(integral * c.dx * c.dy * c.dz);
    }

    Real L1Norm(const VelocityTensor &velocity,
                const VelocityTensor &exact_velocity) {
        double integral = 0.0;
        const Constants &c = velocity.constants;

        // Iterate over the entire tensor space.
        for (std::size_t i = 0; i < c.Nx; ++i) {
            for (std::size_t j = 0; j < c.Ny; ++j) {
                for (std::size_t k = 0; k < c.Nz; ++k) {
                    // Compute differences.
                    const Real diff_u = (i < c.Nx-1) ? velocity.u(i, j, k) - exact_velocity.u(i, j, k) : 0.0;
                    const Real diff_v = (j < c.Ny-1) ? velocity.v(i, j, k) - exact_velocity.v(i, j, k) : 0.0;
                    const Real diff_w = (k < c.Nz-1) ? velocity.w(i, j, k) - exact_velocity.w(i, j, k) : 0.0;

                    // Accumulate abs differences.
                    integral += sqrt(diff_u * diff_u + diff_v * diff_v + diff_w * diff_w);
                }
            }
        }
        return integral * c.dx * c.dy * c.dz;
    }

    Real LInfNorm(const VelocityTensor &velocity,
                  const VelocityTensor &exact_velocity)
    {
        Real integral = 0.0;
        const Constants &c = velocity.constants;

        // Iterate over the entire tensor space.
        for (std::size_t i = 0; i < c.Nx; ++i) {
            for (std::size_t j = 0; j < c.Ny; ++j) {
                for (std::size_t k = 0; k < c.Nz; ++k) {
                    // Compute differences.
                    const Real diff_u = (i < c.Nx-1) ? std::abs(velocity.u(i, j, k) - exact_velocity.u(i, j, k)) : 0.0;
                    const Real diff_v = (j < c.Ny-1) ? std::abs(velocity.v(i, j, k) - exact_velocity.v(i, j, k)) : 0.0;
                    const Real diff_w = (k < c.Nz-1) ? std::abs(velocity.w(i, j, k) - exact_velocity.w(i, j, k)) : 0.0;

                    if (diff_u > integral) {
                        integral = diff_u;
                    }
                    if (diff_v > integral) {
                        integral = diff_v;
                    }
                    if (diff_w > integral) {
                        integral = diff_w;
                    }
                }
            }
        }
        return integral;
    }

}