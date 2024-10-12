#include <cmath>
#include <iostream>
#include "BoundaryConditions.h"
#include "InitialConditions.h"
#include "Manufactured.h"
#include "Timestep.h"

// Return an input function f, depending on x,y,z and time, removing its time dependency.
std::function<Real(Real, Real, Real)> function_at_time(const std::function<Real(Real, Real, Real, Real)> &f, Real time) {
    const std::function<Real(Real, Real, Real)> result = 
            [&time, &f](Real x, Real y, Real z) {
                return f(time, x, y, z);
            };
    return result;
}

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
    constexpr Real Re = 40.0;
    constexpr Real final_time = 10.0;
    constexpr unsigned int num_time_steps = 10; 
    const Constants constants(x_size, y_size, z_size, Nx, Ny, Nz, Re, final_time, num_time_steps);

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
    Tensor<> u_buffer3(sizes);
    Tensor<> v_buffer3(sizes);
    Tensor<> w_buffer3(sizes);

    // Set the initial conditions.
    std::cout << "Setting initial conditions." << std::endl;
    apply_initial_conditions_u(u, function_at_time(u_exact, 0.0), constants);
    apply_initial_conditions_v(v, function_at_time(v_exact, 0.0), constants);
    apply_initial_conditions_w(w, function_at_time(w_exact, 0.0), constants);

    // Compute the solution.
    for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {
        std::cout << "Executing time step " << time_step+1 << "/" << num_time_steps << std::endl;

        // Set the boundary conditions.
        const Real current_time = time_step*constants.dt;
        apply_all_dirichlet_bc<VelocityComponent::u>(u, function_at_time(u_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v, function_at_time(v_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w, function_at_time(v_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::u>(u_buffer1, function_at_time(u_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v_buffer1, function_at_time(v_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w_buffer1, function_at_time(v_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::u>(u_buffer2, function_at_time(u_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v_buffer2, function_at_time(v_exact, current_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w_buffer2, function_at_time(v_exact, current_time), constants);

        // Update the solution inside the mesh.
        timestep(u, v, w, 
                u_buffer1, v_buffer1, w_buffer1, 
                u_buffer2, v_buffer2, w_buffer2, 
                u_buffer3, v_buffer3, w_buffer3, 
                function_at_time(forcing_x, current_time), function_at_time(forcing_y, current_time), function_at_time(forcing_z, current_time), 
                constants);
    }

    // TODO: check the error on the solution.
    std::cout << "Checking the solution." << std::endl;

    return 0;
}