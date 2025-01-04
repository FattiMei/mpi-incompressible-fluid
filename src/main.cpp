#include "TestCaseBoundaries.h"
#include "Norms.h"
#include "PressureEquation.h"
#include "Timestep.h"
#include <cassert>
#include <iostream>
#include <mpi.h>

double Reynolds;

// Simple main for the test case with no pressure and exact solution known.
int main(int argc, char *argv[]) {
  using namespace mif;

  // The rank of the current processor is the id of the processor.
  int rank;
  
  // The size of the communicator is the number of processors overall.
  int size;

  // Initialize MPI.
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // No arguments.
  assert(argc == 1);

  // Input parameters. TODO: read them from a file.
  const size_t Nx_global = 32;
  const size_t Ny_global = 32;
  const size_t Nz_global = 32;
  const Real dt = 1e-4;
  const unsigned int num_time_steps = 1;
  const int Pz = 3;
  const int Py = 2;
  const bool test_case_2 = false;

  // Check processor consistency.
  assert(Pz * Py == size);
  assert(Pz > 0 && Py > 0);

  // Set test case domain information.
  const Real min_x_global = test_case_2 ? -0.5 : 0.0;
  const Real min_y_global = test_case_2 ? -0.5 : 0.0;
  const Real min_z_global = test_case_2 ? -0.5 : -1.0;
  constexpr Real x_size = 1.0;
  constexpr Real y_size = 1.0;
  const Real z_size = test_case_2 ? 1.0 : 2.0;
  constexpr Real Re = 1e3;
  const std::array<bool, 3> periodic_bc{false, false, test_case_2};
  
  // Note that the constants object is only constant within the scope of this
  // particular processor. All processors will have their own subdomain
  // on which they will compute the solution.
  const Constants constants(Nx_global, Ny_global, Nz_global, 
                            x_size, y_size, z_size, 
                            min_x_global, min_y_global, min_z_global,
                            Re, dt*num_time_steps, num_time_steps, 
                            Py, Pz, rank, periodic_bc);
  PressureSolverStructures structures(constants);

  Reynolds = Re;

  // Create the tensors.
  VelocityTensor velocity(constants);
  VelocityTensor velocity_buffer(constants);
  VelocityTensor velocity_buffer_2(constants);
  StaggeredTensor pressure(constants, StaggeringDirection::none);
  StaggeredTensor pressure_buffer(constants, StaggeringDirection::none);
  PressureTensor pressure_solver_buffer(structures);

  // Set the initial conditions.
  TimeVectorFunction exact_velocity(test_case_2 ? exact_u_t2 : exact_u_t1, test_case_2 ? exact_v_t2 : exact_v_t1, test_case_2 ? exact_w_t2 : exact_w_t1);
  velocity.set(exact_velocity.set_time(0.0), true);
  pressure.set(test_case_2 ? exact_p_initial_t2 : exact_p_initial_t1, true);

  // Compute and print convergence conditions.
  Real highest_velocity = 1.0;
  const Real space_step = std::min({constants.dx, constants.dy, constants.dz});
  if (rank == 0) {
    std::cout << "CFL: " << constants.dt / space_step * highest_velocity << std::endl;
    std::cout << "Reynolds condition: " << constants.dt / (Re*space_step*space_step) << std::endl;
  }

  // Compute the solution.
  for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {

    // Set the boundary conditions.
    const Real current_time = time_step * constants.dt;

    // Update the solution inside the mesh.
    timestep(velocity, velocity_buffer, velocity_buffer_2, exact_velocity, current_time, pressure, pressure_buffer, pressure_solver_buffer);
  }

  // TODO: store the required parts of the solution as a vtk and some dat files.

  // Finalize MPI.
  MPI_Finalize();
}
