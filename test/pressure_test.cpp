#include <cmath>
#include "ManufacturedPressure.h"
#include "Norms.h"
#include "PressureEquation.h"

int main(int argc, char* argv[]) {
    using namespace mif;

    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set parameters.
    constexpr Real x_size = 2 * M_PI;
    constexpr Real y_size = x_size;
    constexpr Real z_size = x_size;
    const size_t Nx_domains_global = std::atol(argv[1]);
    const size_t Ny_domains_global = Nx_domains_global;
    const size_t Nz_domains_global = Nx_domains_global;
    constexpr Real time = 1.0;

    const int Pz = std::atol(argv[2]);
    const int Py = size / Pz;
    assert(Pz * Py == size);
    assert(Pz > 0 && Py > 0);

    // Create the needed objects.
    const Constants constants(Nx_domains_global, Ny_domains_global, Nz_domains_global, 
                              x_size, y_size, z_size, 1.0, 1.0, 1, Py, Pz, rank);
    
    VelocityTensor velocity(constants);
    StaggeredTensor pressure({constants.Nx, constants.Ny, constants.Nz}, constants);
    StaggeredTensor pressure_buffer({constants.Nx, constants.Ny, constants.Nz}, constants);
    StaggeredTensor rhs_buffer({constants.Nx, constants.Ny, constants.Nz}, constants);
    StaggeredTensor rhs_buffer2({constants.Nx, constants.Ny, constants.Nz}, constants);

    // Set the right-hand side.
    TimeVectorFunction exact_velocity(u_exact_p_test, v_exact_p_test, w_exact_p_test);
    velocity.set_initial(exact_velocity.set_time(time));

    solve_pressure_equation_neumann(pressure, pressure_buffer, velocity, rhs_buffer, rhs_buffer2);

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
        std::cout << error_l1_global << " " << error_l2_global << " " << error_lInf_global << std::endl;
    }

    MPI_Finalize();
    return 0;
}