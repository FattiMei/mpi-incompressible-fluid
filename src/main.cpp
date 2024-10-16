#include <cassert>
#include <cmath>
#include <iostream>
#include "FunctionHelpers.h"
#include "Manufactured.h"
#include "Norms.h"
#include "Timestep.h"

double Reynolds;

// Simple main for the test case with no pressure and exact solution known.
int main(int argc, char* argv[]) {
    using namespace mif;

    assert(argc == 3);

    // Parameters.
    constexpr Real x_size = 1.0;
    constexpr Real y_size = 1.0;
    constexpr Real z_size = 1.0;
    const size_t Nx = std::atol(argv[1]) + 1UL;
    const size_t Ny = Nx;
    const size_t Nz = Ny;
    constexpr Real Re = 1e6;
    constexpr Real final_time = 1e-4;
    const unsigned int num_time_steps = std::atoi(argv[2]); 
    const Constants constants(Nx, Ny, Nz, x_size, y_size, z_size, Re, final_time, num_time_steps);

    Reynolds = Re;

    // Create the velocity tensors.
    std::array<size_t, 3> u_sizes{Nx-1, Ny, Nz};
    std::array<size_t, 3> v_sizes{Nx, Ny-1, Nz};
    std::array<size_t, 3> w_sizes{Nx, Ny, Nz-1};
    Tensor<> u(u_sizes);
    Tensor<> v(v_sizes);
    Tensor<> w(w_sizes);
    Tensor<> u_buffer1(u_sizes);
    Tensor<> v_buffer1(v_sizes);
    Tensor<> w_buffer1(w_sizes);
    Tensor<> u_buffer2(u_sizes);
    Tensor<> v_buffer2(v_sizes);
    Tensor<> w_buffer2(w_sizes);

    // Set the initial conditions.
    discretize_function<VelocityComponent::u>(u, function_at_time(u_exact, 0.0), constants);
    discretize_function<VelocityComponent::v>(v, function_at_time(v_exact, 0.0), constants);
    discretize_function<VelocityComponent::w>(w, function_at_time(w_exact, 0.0), constants);

    // Compute the solution.
    for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {

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

    // Compute the exact solution.
    Tensor<> u_exact_tensor(u_sizes);
    Tensor<> v_exact_tensor(v_sizes);
    Tensor<> w_exact_tensor(w_sizes);
    discretize_function<VelocityComponent::u>(u_exact_tensor, function_at_time(u_exact, final_time), constants);
    discretize_function<VelocityComponent::v>(v_exact_tensor, function_at_time(v_exact, final_time), constants);
    discretize_function<VelocityComponent::w>(w_exact_tensor, function_at_time(w_exact, final_time), constants);

    // Compute the norms of the error.
    std::cout << L1Norm(u, v, w, u_exact_tensor, v_exact_tensor, w_exact_tensor, constants) << " " << 
                 L2Norm(u, v, w, u_exact_tensor, v_exact_tensor, w_exact_tensor, constants) << " " << 
                 LInfNorm(u, v, w, u_exact_tensor, v_exact_tensor, w_exact_tensor, constants) << std::endl;

    return 0;
}
