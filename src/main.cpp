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

  int rank;
  int size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(argc == 4);

  // Parameters.
  constexpr Real x_size = 1.0;
  constexpr Real y_size = 1.0;
  constexpr Real z_size = 1.0;
  const size_t Nx_domains_global = std::atol(argv[1]);
  const size_t Ny_domains_global = Nx_domains_global;
  const size_t Nz_domains_global = Nx_domains_global;
  constexpr Real Re = 1e4;
  constexpr Real final_time = 1e-4;
  const unsigned int num_time_steps = std::atoi(argv[2]);

  const int Px = std::atol(argv[3]);
  const int Py = size / Px;
  assert(Px * Py == size);
  assert(Px > 0 && Py > 0);
  
  const Constants constants(Nx_domains_global, Ny_domains_global, Nz_domains_global, 
                            x_size, y_size, z_size, Re, final_time, num_time_steps, 
                            Px, Py, rank);

  Reynolds = Re;

  if (rank == 4) {
    std::cout << "Number of processors: " << size << std::endl;
    std::cout << "My rank: " << rank << std::endl;
    std::cout << "My position (x,y): " << constants.x_rank << " " << constants.y_rank << std::endl;
    std::cout << "Neighbors (lowx, highx, lowy, highy): " << constants.prev_x_proc << " " << constants.next_x_proc << " " << constants.prev_y_proc << " " << constants.next_y_proc << std::endl;
    std::cout << "Global sizes (x,y,z): " << Nx_domains_global << " " << Ny_domains_global << " " << Nz_domains_global << std::endl;
    std::cout << "Local sizes (x,y,z): " << constants.Nx_domains_local << " " << constants.Ny_domains_local << " " << constants.Nz_domains << std::endl;
    std::cout << "Local domain x: " << constants.min_x << " " << constants.max_x << std::endl;
    std::cout << "Local domain y: " << constants.min_y << " " << constants.max_y << std::endl;
    std::cout << "Local domain z: " << 0 << " " << constants.z_size << std::endl;
  }

  // Create the velocity tensors.
  VelocityTensor velocity(constants);
  VelocityTensor velocity_buffer(constants);
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
    timestep(velocity, velocity_buffer, rhs_buffer, current_time);
  }

  // Compute the norms of the error.
  std::cout << constants.x_rank << " " << constants.y_rank << " "
            << ErrorL1Norm(velocity, final_time) << " "
            << ErrorL2Norm(velocity, final_time) << " "
            << ErrorLInfNorm(velocity, final_time) << std::endl;

  MPI_Finalize();
}
