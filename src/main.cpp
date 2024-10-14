#include <cmath>
#include <iostream>
#include "FunctionHelpers.h"
#include "Manufactured.h"
#include "Timestep.h"

double Reynolds;

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
    constexpr Real Re = 1000.0;
    constexpr Real final_time = 0.005;
    constexpr unsigned int num_time_steps = 1; 
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

    // TODO: Compute the L2 norm of the error.
    std::cout << "L2 norm of the error: UNIMPLEMENTED" << std::endl;

    // While there is no L2 norm, use a custom norm
    Real u_error = 0.0;
    Real v_error = 0.0;
    Real w_error = 0.0;
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            for (size_t k = 0; k < Nz; k++) {
                const Real u_difference = u(i, j, k) - u_exact_tensor(i, j, k);
                u_error += u_difference * u_difference;
                const Real v_difference = v(i, j, k) - v_exact_tensor(i, j, k);
                v_error += v_difference * v_difference;
                const Real w_difference = w(i, j, k) - w_exact_tensor(i, j, k);
                w_error += w_difference * w_difference;
            }
        }
    }
    u_error = std::sqrt(u_error) / (Nx * Ny * Nz);
    v_error = std::sqrt(v_error) / (Nx * Ny * Nz);
    w_error = std::sqrt(w_error) / (Nx * Ny * Nz);

    std::cout << "Error on u: " << u_error << std::endl;
    std::cout << "Error on v: " << v_error << std::endl;
    std::cout << "Error on w: " << w_error << std::endl;

    return 0;
}
