#include "Manufactured.h"
#include "Norms.h"
#include "Timestep.h"
#include <cassert>
#include <iostream>

double Reynolds;

// Simple main for the test case with no pressure and exact solution known.
int main(int argc, char *argv[]) {
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
  const Constants constants(Nx, Ny, Nz, x_size, y_size, z_size, Re, final_time,
                            num_time_steps);

  Reynolds = Re;

  // Create the velocity tensors.
  VelocityTensor velocity(constants);
  VelocityTensor velocity_buffer1(constants);
  VelocityTensor rhs_buffer(constants);

  // Set the initial conditions.
  TimeVectorFunction exact_velocity(u_exact, v_exact, w_exact);
  velocity.set(exact_velocity.set_time(0.0), true);

  // Compute the solution.
  TimeVectorFunction forcing_term(forcing_x, forcing_y, forcing_z);
  for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {

    // Set the boundary conditions.
    const Real current_time = time_step * constants.dt;

    // Update the solution inside the mesh.
    timestep(velocity, velocity_buffer1, rhs_buffer, current_time);
  }

  // Compute the norms of the error.
  std::cout << ErrorL1Norm(velocity, final_time) << " "
            << ErrorL2Norm(velocity, final_time) << " "
            << ErrorLInfNorm(velocity, final_time) << std::endl;

  return 0;
}
