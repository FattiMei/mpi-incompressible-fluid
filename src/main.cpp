#include "Manufactured.h"
#include "Norms.h"
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

  // Argument 1:
  // Number of domains in the x direction globally.
  // Argument 2:
  // Number of time steps.
  // Argument 3:
  // Number of processors in the x direction.
  assert(argc == 4);

  // Parameters.
  constexpr Real min_x_global = -1.0;
  constexpr Real min_y_global = -1.0;
  constexpr Real min_z_global = -1.0;
  constexpr Real x_size = 2.0;
  constexpr Real y_size = 2.0;
  constexpr Real z_size = 2.0;
  const size_t Nx_domains_global = std::atol(argv[1]);
  const size_t Ny_domains_global = Nx_domains_global;
  const size_t Nz_domains_global = Nx_domains_global;
  constexpr Real Re = 1e4;
  constexpr Real final_time = 1e-4;
  const unsigned int num_time_steps = std::atoi(argv[2]);

  // Given the number of processors in the x direction, compute the number of
  // processors in the y direction and verify that the total number of
  // processors is correct.
  const int Pz = std::atol(argv[3]);
  const int Py = size / Pz;
  assert(Pz * Py == size);
  assert(Pz > 0 && Py > 0);
  
  // Note that the constants object is only constant within the scope of this
  // particular processor. All processors will have their own subdomain
  // on which they will compute the solution.
  const Constants constants(Nx_domains_global, Ny_domains_global, Nz_domains_global, 
                            x_size, y_size, z_size, 
                            min_x_global, min_y_global, min_z_global,
                            Re, final_time, num_time_steps, 
                            Py, Pz, rank);

  Reynolds = Re;

  // Create the tensors.
  VelocityTensor velocity(constants);
  VelocityTensor velocity_buffer(constants);
  VelocityTensor rhs_buffer(constants);
  StaggeredTensor pressure({constants.Nx, constants.Ny, constants.Nz}, constants);
  StaggeredTensor pressure_buffer({constants.Nx, constants.Ny, constants.Nz}, constants);
  PressureSolverStructures structures(constants);

  // Set the initial conditions.
  TimeVectorFunction exact_velocity(u_exact, v_exact, w_exact);
  velocity.set_initial(exact_velocity.set_time(0.0));
  const std::function<Real(Real, Real, Real)> &initial_pressure = [](Real x, Real y, Real z) { return p_exact(0.0, x, y, z); };
  pressure.set(initial_pressure, true);

  // Compute the solution.
  for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {

    // Set the boundary conditions.
    const Real current_time = time_step * constants.dt;

    // Update the solution inside the mesh.
    timestep(velocity, velocity_buffer, rhs_buffer, exact_velocity, current_time, pressure, pressure_buffer, structures);
  }

  // Remove a constant from the pressure.
  const Real difference = p_exact(final_time, 0, 0, 0) - pressure(0,0,0);
  for (size_t k = 0; k < constants.Nz; k++) {
    for (size_t j = 0; j < constants.Ny; j++) {
        for (size_t i = 0; i < constants.Nx; i++) {
            pressure(i,j,k) -= difference;
        }
    }
  }

  // Compute the norms of the error.
  const Real error_l1_local_v = ErrorL1Norm(velocity, exact_velocity, final_time);
  const Real error_l2_local_v = ErrorL2Norm(velocity, exact_velocity, final_time);
  const Real error_lInf_local_v = ErrorLInfNorm(velocity, exact_velocity, final_time);
  const Real error_l1_local_p = ErrorL1Norm(pressure, p_exact, final_time);
  const Real error_l2_local_p = ErrorL2Norm(pressure, p_exact, final_time);
  const Real error_lInf_local_p = ErrorLInfNorm(pressure, p_exact, final_time);
  
  // The global error is computed by accumulating the errors on the processor
  // with rank 0.
  const Real error_l1_global_v = accumulate_error_mpi_l1(error_l1_local_v, constants);
  const Real error_l2_global_v = accumulate_error_mpi_l2(error_l2_local_v, constants);
  const Real error_lInf_global_v = accumulate_error_mpi_linf(error_lInf_local_v, constants);
  const Real error_l1_global_p = accumulate_error_mpi_l1(error_l1_local_p, constants);
  const Real error_l2_global_p = accumulate_error_mpi_l2(error_l2_local_p, constants);
  const Real error_lInf_global_p = accumulate_error_mpi_linf(error_lInf_local_p, constants);

  if (rank == 0) {
    std::cout << error_l1_global_v << " " << error_l2_global_v << " " << error_lInf_global_v << 
    " " << error_l1_global_p << " " << error_l2_global_p << " " << error_lInf_global_p << std::endl;
  }

  // Finalize MPI.
  MPI_Finalize();
}
