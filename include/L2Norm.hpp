#include <vector>
#include <iostream>
#include <cmath>
#include "Tensor.h"

#ifndef MPI_INCOMPRESSIBLE_FLUID_L2Norm
#define MPI_INCOMPRESSIBLE_FLUID_L2Norm
#define Real double


class Constants {
        public:
            // Core constants.
            const Real x_size;
            const Real y_size;
            const Real z_size;
            const size_t Nx;
            const size_t Ny;
            const size_t Nz;
            const Real Re;

            // Derived constants (computed here once for efficiency).
            const Real dx;
            const Real dy;
            const Real dz;
            const size_t row_size;
            const size_t matrix_size;
            const Real one_over_2_dx;
            const Real one_over_2_dy;
            const Real one_over_2_dz;
            const Real one_over_8_dx;
            const Real one_over_8_dy;
            const Real one_over_8_dz;
            const Real one_over_dx2_Re;
            const Real one_over_dy2_Re;
            const Real one_over_dz2_Re;

            // Constructor.
            Constants(size_t Nx, size_t Ny, size_t Nz, Real x_size, Real y_size, Real z_size, Real Re)
        : Nx(Nx), Ny(Ny), Nz(Nz), x_size(x_size), y_size(y_size), z_size(z_size), Re(Re),
          dx(x_size / Nx), dy(y_size / Ny), dz(z_size / Nz),
          row_size(Nx), matrix_size(Nx * Ny),
          one_over_2_dx(1 / (2 * dx)), one_over_2_dy(1 / (2 * dy)), one_over_2_dz(1 / (2 * dz)),
          one_over_8_dx(1 / (8 * dx)), one_over_8_dy(1 / (8 * dy)), one_over_8_dz(1 / (8 * dz)),
          one_over_dx2_Re(1 / (Re * dx * dx)), one_over_dy2_Re(1 / (Re * dy * dy)), one_over_dz2_Re(1 / (Re * dz * dz))
    {
    }
};


/// @brief Computes the L2 norm of the difference between numerical and exact solutions for 3D velocity components (U, V, W).
/// This function calculates the weighted L2 norm by summing the squared differences between the numerical and exact 
/// velocity components for all grid points, using the trapezoidal method, that is second order accuracy.
/// 
/// @param U A vector of numerical values for the velocity component U.
/// @param V A vector of numerical values for the velocity component V.
/// @param W A vector of numerical values for the velocity component W.
/// @param Uex A vector of exact solution values for the velocity component U.
/// @param Vex A vector of exact solution values for the velocity component V.
/// @param Wex A vector of exact solution values for the velocity component W.
/// @param c A `Constants` structure containing grid dimensions (Nx, Ny, Nz) and grid spacing (dx, dy, dz).
/// @return The computed L2 norm, which represents the error between the numerical and exact solutions.
double L2Norm(std::vector<double> &U, std::vector<double> &V, std::vector<double> &W, std::vector<double> &Uex, std::vector<double> &Vex, std::vector<double> &Wex, Constants c)
{
    std::vector<double> wx(c.Nx), wy(c.Ny), wz(c.Nz);
    double integral = 0.0; 

    for (size_t i = 0; i < c.Nx; ++i) {
        wx[i] = (i == 0 || i == c.Nx - 1) ? 1.0 : 0.5;
    }
    for (size_t j = 0; j < c.Ny; ++j) {
        wy[j] = (j == 0 || j == c.Ny - 1) ? 1.0 : 0.5;
    }
    for (size_t k = 0; k < c.Nz; ++k) {
        wz[k] = (k == 0 || k == c.Nz - 1) ? 1.0 : 0.5;
    }

    // Loop over all grid points using a single loop
    for (size_t index = 0; index < U.size(); ++index) {
        // Compute 3D indices from the flat index
        size_t i = index % c.Nx;
        size_t j = (index / c.Nx) % c.Ny;
        size_t k = index / (c.Nx * c.Ny);

        // Retrieve weights
        double weight = wx[i] * wy[j] * wz[k];

        // Compute the differences between numerical and exact velocities
        double diff_u = U[index] - Uex[index];
        double diff_v = V[index] - Vex[index];
        double diff_w = W[index] - Wex[index];

        // Compute the squared difference
        double diff_squared = diff_u * diff_u + diff_v * diff_v + diff_w * diff_w;

        // Accumulate the weighted squared differences
        integral += weight * diff_squared;
    }

    // Multiply by the volume element
    integral *= c.dx * c.dy * c.dz;

    // Take the square root to get the L2 norm
    return std::sqrt(integral);
}

// This one is the same method as before, but with input the Tensor class, declared into "Tensor.h"
// NB: To be tested
template <typename Type, std::size_t SpaceDim>
double L2NormTensor(mif::Tensor<Type, SpaceDim> &U, mif::Tensor<Type, SpaceDim> &V, 
              mif::Tensor<Type, SpaceDim> &W, mif::Tensor<Type, SpaceDim> &Uex, 
              mif::Tensor<Type, SpaceDim> &Vex, mif::Tensor<Type, SpaceDim> &Wex, 
              const Constants &c)
{
    double wxi, wyj, wzk;
    double weight = 0;
    double integral = 0.0;

    // Iterate over the entire tensor space
    for (std::size_t i = 0; i < c.Nx; ++i) {
        wxi = (i == 0 || i == c.Nx - 1) ? 1.0 : 0.5;
        for (std::size_t j = 0; j < c.Ny; ++j) {
            wyj = (j == 0 || j == c.Ny - 1) ? 1.0 : 0.5;
            for (std::size_t k = 0; k < c.Nz; ++k) {
                // Calculate weight for current grid point
                wzk = (k == 0 || k == c.Nz - 1) ? 1.0 : 0.5;

                weight = wxi * wyj * wzk;

                // Compute differences
                double diff_u = U(i, j, k) - Uex(i, j, k);
                double diff_v = V(i, j, k) - Vex(i, j, k);
                double diff_w = W(i, j, k) - Wex(i, j, k);

                // Accumulate squared differences with weights
                integral += weight * (diff_u * diff_u + diff_v * diff_v + diff_w * diff_w);
            }
        }
    }

    // Multiply by volume element and return the square root
    integral *= c.dx * c.dy * c.dz;
    return std::sqrt(integral);
};



#endif //MPI_INCOMPRESSIBLE_FLUID_H