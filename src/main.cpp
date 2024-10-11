//
#include "L2Norm.hpp"

int main()
{
    std::cout<<"Bella li"<<std::endl;
    size_t Nx = 3;
    size_t Ny = 4; 
    size_t Nz = 5; 
    Real x_size = 10.0; 
    Real y_size = 200.0; 
    Real z_size = 1.0; 
    Real Re = 1.0;
    Constants cc(Nx, Ny, Nz, x_size, y_size, z_size, Re);

    size_t totalSize = Nx * Ny * Nz;

    std::vector<double> U(totalSize, 0.0); 
    std::vector<double> V(totalSize, 0.0); 
    std::vector<double> W(totalSize, 0.0); 
    std::vector<double> Uex(totalSize, 0.0); 
    std::vector<double> Vex(totalSize, 0.0); 
    std::vector<double> Wex(totalSize, 0.0);

    const double pi = M_PI;

    // Populate exact solutions and computed solutions
    for (size_t k = 0; k < Nz; ++k) {
        double z = k * cc.dz;
        for (size_t j = 0; j < Ny; ++j) {
            double y = j * cc.dy;
            for (size_t i = 0; i < Nx; ++i) {
                double x = i * cc.dx;
                size_t index = i + j * cc.row_size + k * cc.matrix_size;

                Uex[index] = sin(pi * x) * sin(pi * y) * sin(pi * z);
                Vex[index] = Uex[index];
                Wex[index] = Uex[index];

                U[index] = Uex[index];
                V[index] = Vex[index];
                W[index] = Wex[index];
            }
        }
    }
    //Introduce perturbation at center point
    size_t center_index = 1 + 1 * cc.row_size + 1 * cc.matrix_size;
    U[center_index] += 0.1;
    V[center_index] += 0.1;
    W[center_index] += 0.1;
    

    double res = L2Norm(U, V, W, Uex, Vex, Wex, cc);

    
    std::cout<<"Res = "<< res <<std::endl;

    // Hand calculation for center point
    // double computed_rhs_u = mif::calculate_momentum_rhs_u(u, v, w, cc, center_index);
    // double exact_rhs_u = mif::calculate_momentum_rhs_u(exact_u, exact_v, exact_w, cc, center_index);
    // double residual_u = computed_rhs_u - exact_rhs_u;

    // std::cout << "Residual at center point for u-component: " << residual_u << std::endl;

    // // Compare with hand-calculated value
    // double residual_squared = residual_u * residual_u * cc.dx * cc.dy * cc.dz;
    // double residual_l2_norm_hand = std::sqrt(residual_squared);

    // std::cout << "Hand-calculated Residual L2 Norm at center point: " << residual_l2_norm_hand << std::endl;
    return 0;
}