#include "Norms.h"
#include <cmath>



namespace mif {

    Real L2Norm(const Tensor<> &U, const Tensor<> &V, 
                const Tensor<> &W, const Tensor<> &Uex, 
                const Tensor<> &Vex, const Tensor<> &Wex, 
                const Constants &c) {
        Real integral = 0.0;

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
                    const Real diff_u = U(i, j, k) - Uex(i, j, k);
                    const Real diff_v = V(i, j, k) - Vex(i, j, k);
                    const Real diff_w = W(i, j, k) - Wex(i, j, k);

                    // Accumulate squared differences with weights.
                    integral += weight * (diff_u * diff_u + diff_v * diff_v + diff_w * diff_w);
                }
            }
        }

        // Multiply by volume element and return the square root.
        return std::sqrt(integral * c.dx * c.dy * c.dz);
    }

    Real L1Norm(const Tensor<> &U, const Tensor<> &V, 
                const Tensor<> &W, const Tensor<> &Uex, 
                const Tensor<> &Vex, const Tensor<> &Wex, 
                const mif::Constants &c) {
        Real integral = 0.0;
        // Iterate over the entire tensor space.
        for (std::size_t i = 0; i < c.Nx; ++i) {
            for (std::size_t j = 0; j < c.Ny; ++j) {
                for (std::size_t k = 0; k < c.Nz; ++k) {
                    // Compute differences.
                    const Real diff_u = U(i, j, k) - Uex(i, j, k);
                    const Real diff_v = V(i, j, k) - Vex(i, j, k);
                    const Real diff_w = W(i, j, k) - Wex(i, j, k);

                    // Accumulate abs differences.
                    integral += sqrt(diff_u * diff_u + diff_v * diff_v + diff_w * diff_w);
                }
            }
        }
        return integral * c.dx * c.dy * c.dz;
    }

    Real LInfNorm(const Tensor<> &U, const Tensor<> &V, 
                  const Tensor<> &W, const Tensor<> &Uex, 
                  const Tensor<> &Vex, const Tensor<> &Wex, 
                  const Constants &c)
    {
        Real integral = 0.0;
        // Iterate over the entire tensor space.
        for (std::size_t i = 0; i < c.Nx; ++i) {
            for (std::size_t j = 0; j < c.Ny; ++j) {
                for (std::size_t k = 0; k < c.Nz; ++k) {
                    // Compute differences.
                    const Real diff_u = std::abs(U(i, j, k) - Uex(i, j, k));
                    const Real diff_v = std::abs(V(i, j, k) - Vex(i, j, k));
                    const Real diff_w = std::abs(W(i, j, k) - Wex(i, j, k));

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
