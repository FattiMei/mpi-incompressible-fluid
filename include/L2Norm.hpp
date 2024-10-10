#include <vector>
#include <iostream>
#include <cmath>

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

inline Real calculate_momentum_rhs_u(const std::vector<double> &u, const std::vector<double> &v, const std::vector<double> &w, const Constants &constants, const size_t index) {
    return -u[index]*(u[index+1]-u[index-1])*constants.one_over_2_dx 
            -(v[index]+v[index-constants.row_size]+v[index+1]+v[index+1-constants.row_size])*(u[index+constants.row_size]-u[index-constants.row_size])*constants.one_over_8_dy
            -(w[index]+w[index-constants.matrix_size]+w[index+1]+w[index+1-constants.matrix_size])*(u[index+constants.matrix_size]-u[index-constants.matrix_size])*constants.one_over_8_dz
            +(u[index+1]-2*u[index]+u[index-1])*constants.one_over_dx2_Re
            +(u[index+constants.row_size]-2*u[index]+u[index-constants.row_size])*constants.one_over_dy2_Re
            +(u[index+constants.matrix_size]-2*u[index]+u[index-constants.matrix_size])*constants.one_over_dz2_Re;
}

    // Calculate the y component of -(u*nabla)u + 1/Re nabla^2 u at index index.
inline Real calculate_momentum_rhs_v(const std::vector<double> &u, const std::vector<double> &v, const std::vector<double> &w, const Constants &constants, const size_t index) {
    return -(u[index]+u[index-1]+u[index+constants.row_size]+u[index+constants.row_size-1])*(v[index+1]-v[index-1])*constants.one_over_8_dx
            -v[index]*(v[index+constants.row_size]-v[index-constants.row_size])*constants.one_over_2_dy 
            -(w[index]+w[index-constants.matrix_size]+w[index+constants.row_size]+w[index+constants.row_size-constants.matrix_size])*(v[index+constants.matrix_size]-v[index-constants.matrix_size])*constants.one_over_8_dz
            +(v[index+1]-2*v[index]+v[index-1])*constants.one_over_dx2_Re
            +(v[index+constants.row_size]-2*v[index]+v[index-constants.row_size])*constants.one_over_dy2_Re
            +(v[index+constants.matrix_size]-2*v[index]+v[index-constants.matrix_size])*constants.one_over_dz2_Re;
}

// Calculate the z component of -(u*nabla)u + 1/Re nabla^2 u at index index.
inline Real calculate_momentum_rhs_w(const std::vector<double> &u, const std::vector<double> &v, const std::vector<double> &w, const Constants &constants, const size_t index) {
    return -(u[index]+u[index-1]+u[index+constants.matrix_size]+u[index+constants.matrix_size-1])*(w[index+1]-w[index-1])*constants.one_over_8_dx
            -(v[index]+v[index-constants.row_size]+v[index+constants.matrix_size]+v[index+constants.matrix_size-constants.row_size])*(w[index+constants.row_size]-w[index-constants.row_size])*constants.one_over_8_dy
            -w[index]*(w[index+constants.matrix_size]-w[index-constants.matrix_size])*constants.one_over_2_dz
            +(w[index+1]-2*w[index]+w[index-1])*constants.one_over_dx2_Re
            +(w[index+constants.row_size]-2*w[index]+w[index-constants.row_size])*constants.one_over_dy2_Re
            +(w[index+constants.matrix_size]-2*w[index]+w[index-constants.matrix_size])*constants.one_over_dz2_Re;
}



//template<typename T>
double L2Norm(std::vector<double> &U, std::vector<double> &V, std::vector<double> &W, std::vector<double> &Uex, std::vector<double> &Vex, std::vector<double> &Wex, Constants c){
    double integral = 0.0; 

    for (size_t i = 1; i < c.Nx - 1; ++i) { 
        for (size_t j = 1; j < c.Ny - 1; ++j) { 
            for (size_t k = 1; k < c.Nz - 1; ++k) { 
                size_t index = i + j * c.row_size + k * c.matrix_size; 
 
                // Compute the RHS for computed velocities 
                Real computed_rhs_u = calculate_momentum_rhs_u(U, V, W, c, index); 
                Real computed_rhs_v = calculate_momentum_rhs_v(U, V, W, c, index); 
                Real computed_rhs_w = calculate_momentum_rhs_w(U, V, W, c, index); 
 
                // Compute the RHS for exact velocities 
                Real exact_rhs_u = calculate_momentum_rhs_u(Uex, Vex, Wex, c, index); 
                Real exact_rhs_v = calculate_momentum_rhs_v(Uex, Vex, Wex, c, index); 
                Real exact_rhs_w = calculate_momentum_rhs_w(Uex, Vex, Wex, c, index); 
 
                // Compute the residuals 
                double residual_u = computed_rhs_u - exact_rhs_u; 
                double residual_v = computed_rhs_v - exact_rhs_v; 
                double residual_w = computed_rhs_w - exact_rhs_w; 
 
                // Compute the squared residual 
                double residual_squared = residual_u * residual_u + residual_v * residual_v + residual_w * residual_w; 
 
                // Accumulate the residual (no weighting needed for interior points) 
                integral += residual_squared; 
            } 
        } 
    } 
 
    integral *= c.dx * c.dy * c.dz; 
    return std::sqrt(integral); 
}



#endif //MPI_INCOMPRESSIBLE_FLUID_std::vector<double>_H