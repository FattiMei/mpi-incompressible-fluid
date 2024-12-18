#include <chrono>
#include <cmath>
#include "ManufacturedPressure.h"
#include "Norms.h"
#include "PressureEquation.h"

// Pressure equation test for homogeneous Neumann boundary conditions.
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
    const size_t Nx_domains_global = std::atol(argv[1]);
    const size_t Ny_domains_global = Nx_domains_global * 3;
    const size_t Nz_domains_global = Nx_domains_global * 5;
    constexpr Real time = 1.0;

    const int Pz = std::atol(argv[2]);
    const int Py = size / Pz;
    assert(Pz * Py == size);
    assert(Pz > 0 && Py > 0);

    // Create the needed objects.
    const Constants constants(Nx_domains_global, Ny_domains_global, Nz_domains_global, 
                              x_size, y_size, z_size, 
                              min_x_global, min_y_global, min_z_global,
                              1.0, 1.0, 1, Py, Pz, rank);
    
    VelocityTensor velocity(constants);
    StaggeredTensor pressure({constants.Nx, constants.Ny, constants.Nz}, constants);

    PressureSolverStructures structures(constants);

    // Set the right-hand side.
    TimeVectorFunction exact_velocity(u_exact_p_test, v_exact_p_test, w_exact_p_test);
    velocity.set(exact_velocity.set_time(time), true);

    // Solve.
    const auto before = chrono::high_resolution_clock::now();
    solve_pressure_equation_homogeneous_neumann(pressure, velocity, structures);
    const auto after = chrono::high_resolution_clock::now();
    const Real execution_time = (after-before).count() / 1e9;

    // Remove a constant.
    const Real difference = p_exact_p_test(time, min_x_global, min_y_global, min_z_global) - pressure(0,0,0);
    for (size_t k = 0; k < constants.Nz; k++) {
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                pressure(i,j,k) += difference;
            }
        }
    }

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
        std::cout << "Time: " << execution_time << "s " << execution_time/Nx_domains_global/Ny_domains_global/Nz_domains_global << std::endl;
        std::cout << "Errors: " << error_l1_global << " " << error_l2_global << " " << error_lInf_global << std::endl;
    }

    MPI_Finalize();
    return 0;
}