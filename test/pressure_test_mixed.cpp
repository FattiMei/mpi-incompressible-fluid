#include <chrono>
#include <cmath>
#include "ManufacturedPressure.h"
#include "Norms.h"
#include "PressureEquation.h"

// Pressure equation test for homogeneous Neumann boundary conditions
// on x,y faces and periodic conditions on the z faces.
int main(int argc, char* argv[]) {
    using namespace mif;

    assert(argc == 3);

    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set parameters.
    constexpr Real min_x_global = 0.0;
    constexpr Real min_y_global = 0.0;
    constexpr Real min_z_global = 0.0;
    constexpr Real x_size = 2 * M_PI;
    constexpr Real y_size = x_size;
    constexpr Real z_size = x_size;
    const size_t Nx_global = std::atol(argv[1]);
    const size_t Ny_global = Nx_global * 3;
    const size_t Nz_global = Nx_global * 5;
    constexpr Real time = 1.0;
    const std::array<bool, 3> periodic_bc{false, false, true};

    const int Pz = std::atol(argv[2]);
    const int Py = size / Pz;
    assert(Pz * Py == size);
    assert(Pz > 0 && Py > 0);

    // Create the needed objects.
    // Note: it is necessary that delta t equals one, as manufsol_pressure.py creates 
    // a velocity field with gradient equal to the divergence of the velocity, without dividing
    // by delta t, as time dependency is not relevant for this test.
    const Constants constants(Nx_global, Ny_global, Nz_global, 
                              x_size, y_size, z_size, 
                              min_x_global, min_y_global, min_z_global,
                              1.0, 1.0, 1, Py, Pz, rank, periodic_bc);
    PressureSolverStructures structures(constants);
    
    VelocityTensor velocity(constants);
    PressureTensor pressure_solver_buffer(structures);
    StaggeredTensor pressure(constants, StaggeringDirection::none);

    // Set the right-hand side.
    TimeVectorFunction exact_velocity(u_exact_p_test, v_exact_p_test, w_exact_p_test);
    velocity.set(exact_velocity.set_time(time), true);

    // Solve.
    const auto before = chrono::high_resolution_clock::now();
    solve_pressure_equation_homogeneous_periodic(pressure, pressure_solver_buffer, velocity, constants.dt);
    const auto after = chrono::high_resolution_clock::now();
    const Real execution_time = (after-before).count() / 1e9;

    // Remove a constant.
    adjust_pressure(pressure, [&time](Real x, Real y, Real z){return p_exact_p_test(time,x,y,z);});

    // Compute the norms of the error.
    const Real error_l1_local = ErrorL1Norm(pressure, p_exact_p_test, time);
    const Real error_l2_local = ErrorL2Norm(pressure, p_exact_p_test, time);
    const Real error_lInf_local = ErrorLInfNorm(pressure, p_exact_p_test, time);

    // The global error is computed by accumulating the errors on the processor
    // with rank 0.
    const Real error_l1_global = accumulate_error_mpi_l1(error_l1_local, constants);
    const Real error_l2_global = accumulate_error_mpi_l2(error_l2_local, constants);
    const Real error_lInf_global = accumulate_error_mpi_linf(error_lInf_local, constants);

    if (rank == 0) {
        std::cout << "Time: " << execution_time << "s " << execution_time/Nx_global/Ny_global/Nz_global << std::endl;
        std::cout << "Errors: " << error_l1_global << " " << error_l2_global << " " << error_lInf_global << std::endl;
    }

    MPI_Finalize();
    return 0;
}