#include <cmath>
#include <iostream>
#include "BoundaryConditions.h"
#include "InitialConditions.h"
#include "Timestep.h"

Real exact_u(Real x, Real y, Real z, Real t) {
    return std::sin(x) * std::cos(y) * std::cos(z) * std::sin(t);
}

Real exact_v(Real x, Real y, Real z, Real t) {
    return std::cos(x) * std::sin(y) * std::sin(z) * std::sin(t);
}

Real exact_w(Real x, Real y, Real z, Real t) {
    return 2 * std::cos(x) * std::cos(y) * std::cos(z) * std::sin(t);
}

Real forcing_term(Real x, Real y, Real z, Real t) {
    // TODO: implement this.
    return 0.0;
}

// Return an input function f, depending on x,y,z and time, removing its time dependency.
std::function<Real(Real, Real, Real)> function_at_time(const std::function<Real(Real, Real, Real, Real)> &f, Real time) {
    const std::function<Real(Real, Real, Real)> result = 
            [&time, &f](Real x, Real y, Real z) {
                return f(x, y, z, time);
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
    Tensor<> u(std::array<size_t, 3>{Nx, Ny, Nz});
    Tensor<> v(std::array<size_t, 3>{Nx, Ny, Nz});
    Tensor<> w(std::array<size_t, 3>{Nx, Ny, Nz});

    // Set the initial conditions.
    std::cout << "Setting initial conditions." << std::endl;
    apply_initial_conditions_u(u, function_at_time(exact_u, 0.0), constants);
    apply_initial_conditions_v(v, function_at_time(exact_v, 0.0), constants);
    apply_initial_conditions_w(w, function_at_time(exact_w, 0.0), constants);

    // Compute the solution.
    for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {
        std::cout << "Executing time step " << time_step+1 << "/" << num_time_steps << std::endl;

        // Create a function for the forcing term at the current time.
        const std::function<Real(Real, Real, Real)> current_forcing_term = 
            [&constants, &time_step](Real x, Real y, Real z) {
                return forcing_term(x, y, z, constants.dt * time_step);
            };

        // Set the boundary conditions.
        apply_all_dirichlet_bc(u, v, w, current_forcing_term, constants);

        // Update the solution inside the mesh.
        timestep(u, v, w, constants);
    }

    // TODO: check the error on the solution.
    std::cout << "Checking the solution." << std::endl;

    return 0;
}