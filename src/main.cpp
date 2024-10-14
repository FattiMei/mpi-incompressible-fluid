#include <cmath>
#include <iostream>
#include "FunctionHelpers.h"
#include "Manufactured.h"
#include "Timestep.h"

double Reynolds;

Real L2Norm(mif::Tensor<> &U, mif::Tensor<> &V, 
            mif::Tensor<> &W, mif::Tensor<> &Uex, 
            mif::Tensor<> &Vex, mif::Tensor<> &Wex, 
            const mif::Constants &c)
{
    double wxi, wyj, wzk;
    double integral = 0.0;

    // Iterate over the entire tensor space
    for (std::size_t i = 0; i < c.Nx; ++i) {
        wxi = (i == 0 || i == c.Nx - 1) ? 0.5 : 1.0;
        for (std::size_t j = 0; j < c.Ny; ++j) {
            wyj = (j == 0 || j == c.Ny - 1) ? 0.5 : 1.0;
            for (std::size_t k = 0; k < c.Nz; ++k) {
                wzk = (k == 0 || k == c.Nz - 1) ? 0.5 : 1.0;

                const double weight = wxi * wyj * wzk;

                // Compute differences
                const double diff_u = U(i, j, k) - Uex(i, j, k);
                const double diff_v = V(i, j, k) - Vex(i, j, k);
                const double diff_w = W(i, j, k) - Wex(i, j, k);

                // Accumulate squared differences with weights
                integral += weight * (diff_u * diff_u + diff_v * diff_v + diff_w * diff_w);
            }
        }
    }

    // Multiply by volume element and return the square root
    return std::sqrt(integral * c.dx * c.dy * c.dz);
};

// Simple main for the test case with no pressure and exact solution known.
int main(int argc, char* argv[]) {
    using namespace mif;

    // Parameters.
    constexpr Real x_size = 1.0;
    constexpr Real y_size = 1.0;
    constexpr Real z_size = 1.0;
    constexpr size_t Nx = 64;
    constexpr size_t Ny = 64;
    constexpr size_t Nz = 64;
    constexpr Real Re = 10000.0;
    constexpr Real final_time = 0.001;
    constexpr unsigned int num_time_steps = 2; 
    const Constants constants(Nx, Ny, Nz, x_size, y_size, z_size, Re, final_time, num_time_steps);

    Reynolds = Re;

    // Create the velocity tensors.
    std::array<size_t, 3> sizes{Nx, Ny, Nz};
    Tensor<> u(sizes);
    Tensor<> v(sizes);
    Tensor<> w(sizes);
    Tensor<> u_buffer1(sizes);
    Tensor<> v_buffer1(sizes);
    Tensor<> w_buffer1(sizes);
    Tensor<> u_buffer2(sizes);
    Tensor<> v_buffer2(sizes);
    Tensor<> w_buffer2(sizes);

    // Set the initial conditions.
    std::cout << "Setting initial conditions." << std::endl;
    discretize_function<VelocityComponent::u>(u, function_at_time(u_exact, 0.0), constants);
    discretize_function<VelocityComponent::v>(v, function_at_time(v_exact, 0.0), constants);
    discretize_function<VelocityComponent::w>(w, function_at_time(w_exact, 0.0), constants);

    // Compute the solution.
    for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {
        std::cout << "Executing time step " << time_step+1 << "/" << num_time_steps << std::endl;

        // Set the boundary conditions.
        const Real current_time = time_step*constants.dt;

        // Update the solution inside the mesh.
        timestep(u, v, w, 
                u_buffer1, v_buffer1, w_buffer1, 
                u_buffer2, v_buffer2, w_buffer2,
                u_exact, v_exact, w_exact,
                forcing_x, forcing_y, forcing_z, 
                current_time, constants);
    }

    // Check the error on the solution.
    std::cout << "Checking the solution." << std::endl;

    // Compute the exact solution.
    Tensor<> u_exact_tensor(sizes);
    Tensor<> v_exact_tensor(sizes);
    Tensor<> w_exact_tensor(sizes);
    discretize_function<VelocityComponent::u>(u_exact_tensor, function_at_time(u_exact, final_time), constants);
    discretize_function<VelocityComponent::v>(v_exact_tensor, function_at_time(v_exact, final_time), constants);
    discretize_function<VelocityComponent::w>(w_exact_tensor, function_at_time(w_exact, final_time), constants);

    // Compute the L2 norm of the error.
    std::cout << "L2 norm of the error: " << L2Norm(u, v, w, u_exact_tensor, v_exact_tensor, w_exact_tensor, constants) << std::endl;

    return 0;
}
