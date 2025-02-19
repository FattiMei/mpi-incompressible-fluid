#include <cassert>
#include <iostream>
#include "ManufacturedVelocity.h"
#include "Norms.h"
#include "TimestepVelocity.h"

double Reynolds;

// Simple main for the test case with no pressure and exact solution known,
// with all Dirichlet BC.
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
  constexpr Real min_x_global = 0.0;
  constexpr Real min_y_global = 0.0;
  constexpr Real min_z_global = 0.0;
  constexpr Real x_size = 1.0;
  constexpr Real y_size = 1.0;
  constexpr Real z_size = 1.0;
  const size_t Nx_global = std::atol(argv[1]);
  const size_t Ny_global = Nx_global;
  const size_t Nz_global = Nx_global;
  constexpr Real Re = 1e4;
  constexpr Real final_time = 1e-4;
  const unsigned int num_time_steps = std::atoi(argv[2]);
  const std::array<bool, 3> periodic_bc{false, false, false};

  // Given the number of processors in the z direction, compute the number of
  // processors in the y direction and verify that the total number of
  // processors is correct.
  const int Pz = std::atol(argv[3]);
  const int Py = size / Pz;
  assert(Pz * Py == size);
  assert(Pz > 0 && Py > 0);
  
  // Note that the constants object is only constant within the scope of this
  // particular processor. All processors will have their own subdomain
  // on which they will compute the solution.
  const Constants constants(Nx_global, Ny_global, Nz_global, 
                            x_size, y_size, z_size, 
                            min_x_global, min_y_global, min_z_global,
                            Re, final_time, num_time_steps, 
                            Py, Pz, rank, periodic_bc);

  Reynolds = Re;

  // Create the velocity tensors.
  VelocityTensor velocity(constants);
  VelocityTensor velocity_buffer(constants);
  VelocityTensor rhs_buffer(constants);

  // Set the initial conditions.
  TimeVectorFunction exact_velocity(u_exact_v_test, v_exact_v_test, w_exact_v_test);
  velocity.set(exact_velocity.set_time(0.0), true);

  // Compute the solution.
  for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {

    // Set the boundary conditions.
    const Real current_time = time_step * constants.dt;

    // Update the solution inside the mesh.
    timestep_velocity(velocity, velocity_buffer, rhs_buffer, exact_velocity, current_time);
  }

  // Compute the norms of the error.
  const Real error_l1_local = ErrorL1Norm(velocity, exact_velocity, final_time);
  const Real error_l2_local = ErrorL2Norm(velocity, exact_velocity, final_time);
  const Real error_lInf_local = ErrorLInfNorm(velocity, exact_velocity, final_time);
  
  // The global error is computed by accumulating the errors on the processor
  // with rank 0.
  const Real error_l1_global = accumulate_error_mpi_l1(error_l1_local, constants);
  const Real error_l2_global = accumulate_error_mpi_l2(error_l2_local, constants);
  const Real error_lInf_global = accumulate_error_mpi_linf(error_lInf_local, constants);

  if (rank == 0) {
    std::cout << error_l1_global << " " << error_l2_global << " " << error_lInf_global << std::endl;
  }

  // Finalize MPI.
  MPI_Finalize();
}
