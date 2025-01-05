#include <cassert>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include "Manufactured.h"
#include "Norms.h"
#include "PressureEquation.h"
#include "PressureGradient.h"
#include "StaggeredTensorMacros.h"
#include "Timestep.h"
#include "VTKExport.h"

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
  constexpr Real min_x_global = 0.0;
  constexpr Real min_y_global = 0.0;
  constexpr Real min_z_global = -1.0;
  constexpr Real x_size = 1.0;
  constexpr Real y_size = 1.0;
  constexpr Real z_size = 2.0;
  const size_t Nx_global = std::atol(argv[1]);
  const size_t Ny_global = Nx_global;
  const size_t Nz_global = Nx_global;
  constexpr Real Re = 1e3;
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
  TimeVectorFunction exact_velocity(u_exact, v_exact, w_exact);
  velocity.set(exact_velocity.set_time(0.0), true);
  const std::function<Real(Real, Real, Real)> &initial_pressure = [](Real x, Real y, Real z) { return p_exact(0.0, x, y, z); };
  pressure.set(initial_pressure, true);
  TimeVectorFunction exact_pressure_gradient(dp_dx_exact, dp_dy_exact, dp_dz_exact);

  // Compute and print convergence conditions.
  // Note: assuming the highest velocity value is obtained at time 0.
  /*
  Real highest_velocity = 0.0;
  for (size_t k = 0; k < constants.Nz; k++) {
    for (size_t j = 0; j < constants.Ny; j++) {
      for (size_t i = 0; i < constants.Nx; i++) {
        if (std::abs(velocity.u(i,j,k)) > highest_velocity) {
          highest_velocity = std::abs(velocity.u(i,j,k));
        }
        if (std::abs(velocity.v(i,j,k)) > highest_velocity) {
          highest_velocity = std::abs(velocity.v(i,j,k));
        }
        if (std::abs(velocity.w(i,j,k)) > highest_velocity) {
          highest_velocity = std::abs(velocity.w(i,j,k));
        }
      }
    }
  }
  const Real space_step = std::min({constants.dx, constants.dy, constants.dz});
  if (rank == 0) {
    std::cout << "CFL: " << constants.dt / space_step * highest_velocity << std::endl;
    std::cout << "Reynolds condition: " << constants.dt / (Re*space_step*space_step) << std::endl;
  }
  */

  // Compute the solution.
  const bool print_times = false;
  const auto before = chrono::high_resolution_clock::now();

  for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {

    // Set the boundary conditions.
    const Real current_time = time_step * constants.dt;

    // Update the solution inside the mesh.
    timestep(velocity, velocity_buffer, velocity_buffer_2, exact_velocity, current_time, pressure, pressure_buffer, pressure_solver_buffer);
  }

  if (print_times) {
    MPI_Barrier(MPI_COMM_WORLD);
    const auto after = chrono::high_resolution_clock::now();
    const Real execution_time = (after-before).count() / 1e9;
    if (rank == 0) {
      std::cout << "Absolute time: " << execution_time << "s" << std::endl;
      std::cout << "Relative time: " << execution_time/Nx_global/Ny_global/Nz_global/num_time_steps << std::endl;
    }
  }

  // Compute the pressure gradient.
  VelocityTensor pressure_gradient(constants);
  VELOCITY_TENSOR_SET_FOR_ALL_POINTS(pressure_gradient, pressure_gradient_u, pressure_gradient_v, pressure_gradient_w, false, pressure, i, j, k)
  pressure_gradient.apply_bc(exact_pressure_gradient.set_time(final_time));

  // Remove a constant from the pressure.
  adjust_pressure(pressure, [&final_time](Real x, Real y, Real z){return p_exact(final_time,x,y,z);});

  // Compute the norms of the error.
  const Real error_l1_local_v = ErrorL1Norm(velocity, exact_velocity, final_time);
  const Real error_l2_local_v = ErrorL2Norm(velocity, exact_velocity, final_time);
  const Real error_lInf_local_v = ErrorLInfNorm(velocity, exact_velocity, final_time);
  const Real error_l1_local_p = ErrorL1Norm(pressure, p_exact, final_time);
  const Real error_l2_local_p = ErrorL2Norm(pressure, p_exact, final_time);
  const Real error_lInf_local_p = ErrorLInfNorm(pressure, p_exact, final_time);
  const Real error_l1_local_pg = ErrorL1Norm(pressure_gradient, exact_pressure_gradient, final_time);
  const Real error_l2_local_pg = ErrorL2Norm(pressure_gradient, exact_pressure_gradient, final_time);
  const Real error_lInf_local_pg = ErrorLInfNorm(pressure_gradient, exact_pressure_gradient, final_time);
  
  // The global error is computed by accumulating the errors on the processor
  // with rank 0.
  const Real error_l1_global_v = accumulate_error_mpi_l1(error_l1_local_v, constants);
  const Real error_l2_global_v = accumulate_error_mpi_l2(error_l2_local_v, constants);
  const Real error_lInf_global_v = accumulate_error_mpi_linf(error_lInf_local_v, constants);
  const Real error_l1_global_p = accumulate_error_mpi_l1(error_l1_local_p, constants);
  const Real error_l2_global_p = accumulate_error_mpi_l2(error_l2_local_p, constants);
  const Real error_lInf_global_p = accumulate_error_mpi_linf(error_lInf_local_p, constants);
  const Real error_l1_global_pg = accumulate_error_mpi_l1(error_l1_local_pg, constants);
  const Real error_l2_global_pg = accumulate_error_mpi_l2(error_l2_local_pg, constants);
  const Real error_lInf_global_pg = accumulate_error_mpi_linf(error_lInf_local_pg, constants);

  if (rank == 0) {
    std::cout << error_l1_global_v << " " << error_l2_global_v << " " << error_lInf_global_v << 
    " " << error_l1_global_p << " " << error_l2_global_p << " " << error_lInf_global_p << 
    " " << error_l1_global_pg << " " << error_l2_global_pg << " " << error_lInf_global_pg << std::endl;
  }

  writeVTK("solution.vtk", velocity, pressure);

  if (rank == 0 && size == 1) {
    writeVTKFullMesh("full.vtk", velocity, pressure);
  }


  //direction is 0 for x, 1 for y, 2 for z. this is the axis witch the line is parallel to
  // x,y,z are the coordinates of the point contained in the line
  writeDat("line1.dat", velocity, constants, pressure, rank, size, 0, 0.5, 0.5, 0.0);

  // Finalize MPI.
  MPI_Finalize();
}
