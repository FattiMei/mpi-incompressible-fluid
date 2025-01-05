#include "InputParser.h"
#include "Norms.h"
#include "PressureEquation.h"
#include "TestCaseBoundaries.h"
#include "Timestep.h"
#include "VTKExport.h"
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

  // One argument: input file path.
  assert(argc == 2);
  const std::string input_file = argv[1];

  // Input parameters from the file.
  size_t Nx_global = 0;
  size_t Ny_global = 0;
  size_t Nz_global = 0;
  Real dt = 0;
  unsigned int num_time_steps = 0;
  int Pz = 0;
  int Py = 0;
  bool test_case_2 = 0;
  try {
      // Parse the input file.
      parse_input_file(input_file, Nx_global, Ny_global, Nz_global,
                       dt, num_time_steps, Py, Pz, test_case_2);

      // Check processor consistency.
      if (Pz < 1 || Py < 1) {
        if (rank == 0) std::cerr << "The number of processors in each direction must be at least 1." << std::endl;
        MPI_Finalize();
        return 0;
      }
      if (Pz * Py != size) {
        if (rank == 0) std::cerr << "The number of precessors in the input file do not match with the ones provided to mpirun." << std::endl;
        MPI_Finalize();
        return 0;
      }
  } catch (const std::exception &ex) {
      if (rank == 0) std::cerr << "Error parsing input file: " << ex.what() << std::endl;
      MPI_Finalize();
      return 0;
  }

  // Set test case domain information.

  constexpr Real Re = 1e3;
  const std::array<bool, 3> periodic_bc{false, false, test_case_2};
  // Create needed structures.
  // Note that the constants object is only constant within the scope of this
  // particular processor. All processors will have their own subdomain
  // on which they will compute the solution.
  const Constants constants(Nx_global, Ny_global, Nz_global,
                            1.0, 1.0, test_case_2 ? 1.0 : 2.0,
                            test_case_2 ? -0.5 : 0.0, test_case_2 ? -0.5 : 0.0, test_case_2 ? -0.5 : -1.0,
                            Re, dt * num_time_steps, num_time_steps,
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

  // Compute the solution.
  for (unsigned int time_step = 0; time_step < num_time_steps; time_step++) {

    // Set the boundary conditions.
    const Real current_time = time_step * constants.dt;

    // Update the solution inside the mesh.
    timestep(velocity, velocity_buffer, velocity_buffer_2, exact_velocity, current_time, pressure, pressure_buffer, pressure_solver_buffer);
  }

  // Store the requested slices as a VTK file.
  writeVTK("solution.vtk", velocity, pressure);

  if (rank == 0 && size == 1) {
    writeVTKFullMesh("full.vtk", velocity, pressure);
  }

  // Store the required parts of the solution as dat files.
  if (!test_case_2){
    writeDat("profile1.dat", velocity, pressure, 1, 0.5, 0.5, 0);
    writeDat("profile2.dat", velocity, pressure, 0, 0.5, 0.5, 0);
  }
  else{
    writeDat("profile1.dat", velocity, pressure, 1, 0, 0, 0);
    writeDat("profile2.dat", velocity, pressure, 0, 0, 0, 0);
    writeDat("profile3.dat", velocity, pressure, 2, 0, 0, 0);
  }

  // Finalize MPI.
  MPI_Finalize();
}
