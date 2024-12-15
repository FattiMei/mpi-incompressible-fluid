#include "Norms.h"
#include <cmath>
#include <mpi.h>

namespace mif {

// Compute and return a measure of the error on the velocity over the whole
// domain. Depending on the specified reduction_operation, this can be used to
// define the L1, L2 and Linfinity norms.
Real compute_error(
    const VelocityTensor &velocity, const TimeVectorFunction &exact_velocity, Real time,
    const std::function<Real(Real, Real, Real, Real)> &reduction_operation) {
  Real integral = 0.0;
  const Constants &constants = velocity.constants;

  // To obtain a second order approximation of the norms, the components of the
  // velocity are interpolated into pressure points using a second order
  // approximation. A second order scheme to approximate integrals is then
  // applied to the resulting values. Note: we are skipping integration over
  // boundary points because the error on the boundaries is 0 due to Dirichlet
  // boundary conditions. This also implies that the integration weights are all
  // equal.
  for (size_t k = 1; k < constants.Nz - 1; k++) {
    const Real z = constants.min_z + k * constants.dz;
    for (size_t j = 1; j < constants.Ny - 1; j++) {
      const Real y = constants.min_y + j * constants.dy;
      for (size_t i = 1; i < constants.Nx - 1; i++) {
        const Real x = i * constants.dx;

        const Real interpolated_u =
            (velocity.u(i, j, k) + velocity.u(i + 1, j, k)) / 2.0;
        const Real interpolated_v =
            (velocity.v(i, j, k) + velocity.v(i, j + 1, k)) / 2.0;
        const Real interpolated_w =
            (velocity.w(i, j, k) + velocity.w(i, j, k + 1)) / 2.0;
        const Real u_error = exact_velocity.f_u(time, x, y, z) - interpolated_u;
        const Real v_error = exact_velocity.f_v(time, x, y, z) - interpolated_v;
        const Real w_error = exact_velocity.f_w(time, x, y, z) - interpolated_w;

        integral = reduction_operation(integral, u_error, v_error, w_error);
      }
    }
  }

  return integral;
}

Real ErrorL1Norm(const VelocityTensor &velocity, const TimeVectorFunction &exact_velocity, Real time) {
  const Constants &constants = velocity.constants;

  // Accumulate the module of the error.
  const auto reduction_operation = [](Real integral, Real u_error, Real v_error,
                                      Real w_error) {
    return integral +
           std::sqrt(u_error * u_error + v_error * v_error + w_error * w_error);
  };
  const Real integral = compute_error(velocity, exact_velocity, time, reduction_operation);

  // Multiply the integral by the volume of a cell.
  return integral * constants.dx * constants.dy * constants.dz;
}

Real ErrorL2Norm(const VelocityTensor &velocity, const TimeVectorFunction &exact_velocity, Real time) {
  const Constants &constants = velocity.constants;

  // Accumulate the sum of squared error over the components.
  const auto reduction_operation = [](Real integral, Real u_error, Real v_error,
                                      Real w_error) {
    return integral + u_error * u_error + v_error * v_error + w_error * w_error;
  };
  const Real integral = compute_error(velocity, exact_velocity, time, reduction_operation);

  // Multiply the integral by the volume of a cell and return its square root.
  return std::sqrt(integral * constants.dx * constants.dy * constants.dz);
}

Real ErrorLInfNorm(const VelocityTensor &velocity, const TimeVectorFunction &exact_velocity, Real time) {
  // Return the highest error yet.
  const auto reduction_operation = [](Real integral, Real u_error, Real v_error,
                                      Real w_error) {
    return std::max({integral, std::abs(u_error), std::abs(v_error), std::abs(w_error)});
  };
  return compute_error(velocity, exact_velocity, time, reduction_operation);
}

// Same function for a scalar tensor. No need to skip borders since it is unstaggered.
Real compute_error(
    const StaggeredTensor &pressure, const std::function<Real(Real, Real, Real, Real)> &exact_pressure, 
    Real time, const std::function<Real(Real, Real)> &reduction_operation) {
  Real integral = 0.0;
  const Constants &constants = pressure.constants;

  for (size_t k = 0; k < constants.Nz; k++) {
    const Real z = constants.min_z + k * constants.dz;
    for (size_t j = 0; j < constants.Ny; j++) {
      const Real y = constants.min_y + j * constants.dy;
      for (size_t i = 0; i < constants.Nx; i++) {
        const Real x = i * constants.dx;

        const Real error = exact_pressure(time, x, y, z) - pressure(i,j,k);

        integral = reduction_operation(integral, error);
      }
    }
  }

  return integral;
}

Real ErrorL1Norm(const StaggeredTensor &pressure, const std::function<Real(Real, Real, Real, Real)> &exact_pressure, Real time) {
  const Constants &constants = pressure.constants;
  const auto reduction_operation = [](Real integral, Real error) { return integral + std::abs(error); };
  return compute_error(pressure, exact_pressure, time, reduction_operation) * constants.dx * constants.dy * constants.dz;
}

Real ErrorL2Norm(const StaggeredTensor &pressure, const std::function<Real(Real, Real, Real, Real)> &exact_pressure, Real time) {
  const Constants &constants = pressure.constants;
  const auto reduction_operation = [](Real integral, Real error) { return integral + error*error; };
  return std::sqrt(compute_error(pressure, exact_pressure, time, reduction_operation) * constants.dx * constants.dy * constants.dz);
}

Real ErrorLInfNorm(const StaggeredTensor &pressure, const std::function<Real(Real, Real, Real, Real)> &exact_pressure, Real time) {
  const auto reduction_operation = [](Real integral, Real error) { return std::max(integral, std::abs(error)); };
  return compute_error(pressure, exact_pressure, time, reduction_operation);
}

// Accumulate errors by sending them to the processor with rank 0, according to reduction_operation.
// For all other processors, the function will return -1.
Real accumulate_error_mpi(Real local_error, const Constants &constants,
                          const std::function<Real(Real, Real)> &reduction_operation) {
  if (constants.P == 1) {
    return local_error;
  }

  if (constants.rank == 0) {
    Real global_error = local_error;

    for (int rank = 1; rank < constants.P; ++rank) {
      Real other_error;
      MPI_Status status;
      int outcome = MPI_Recv(&other_error, 1, MPI_MIF_REAL, rank, 0, MPI_COMM_WORLD, &status);
      assert(outcome == MPI_SUCCESS);
      (void) outcome;

      global_error = reduction_operation(global_error, other_error);
    }

    return global_error;
  } else {
    int outcome = MPI_Send(&local_error, 1, MPI_MIF_REAL, 0, 0, MPI_COMM_WORLD);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
    return -1;
  }
}

Real accumulate_error_mpi_l1(Real local_error, const Constants &constants) {
  const auto reduction_operation = [](Real local, Real global) { return local + global; };
  return accumulate_error_mpi(local_error, constants, reduction_operation);
}

Real accumulate_error_mpi_l2(Real local_error, const Constants &constants) {
  const auto reduction_operation = [](Real local, Real global) { return std::sqrt(local*local + global*global); };
  return accumulate_error_mpi(local_error, constants, reduction_operation);
}

Real accumulate_error_mpi_linf(Real local_error, const Constants &constants) {
  const auto reduction_operation = [](Real local, Real global) { return std::max({local, global}); };
  return accumulate_error_mpi(local_error, constants, reduction_operation);
}

} // namespace mif