#include "ManufacturedVelocity.h"
#include "Norms.h"
#include "TimestepVelocity.h"
#include <iostream>

double Reynolds = 1e4;
constexpr Real Lx = 1.0;
constexpr Real Ly = 1.0;
constexpr Real Lz = 1.0;
constexpr Real T = 1e-4;

using namespace mif;

int main() {
  std::cout << "deltax,deltat,l1,l2,linf" << std::endl;

  for (int N = 32, ntime_steps = 1; N < 256; N *= 2, ntime_steps *= 2) {
    const size_t Nx = N;
    const size_t Ny = Nx;
    const size_t Nz = Nx;

    const Constants constants(Nx, Ny, Nz, Lx, Ly, Lz, 0.0, 0.0, 0.0, Reynolds, T, ntime_steps, 1, 1, 0);

    VelocityTensor velocity(constants);
    VelocityTensor velocity_buffer(constants);
    VelocityTensor rhs_buffer(constants);

    TimeVectorFunction exact_velocity(u_exact_v_test, v_exact_v_test, w_exact_v_test);
    velocity.set(exact_velocity.set_time(0.0), true);

    TimeVectorFunction forcing_term(forcing_x, forcing_y, forcing_z);

    for (int step = 0; step < ntime_steps; ++step) {
      const Real t = step * constants.dt;

      timestep_velocity(velocity, velocity_buffer, rhs_buffer, exact_velocity, t);
    }

    std::cout << constants.dx << ',' << constants.dt << ','
              << ErrorL1Norm(velocity, exact_velocity, T) << ',' << ErrorL2Norm(velocity, exact_velocity, T)
              << ',' << ErrorLInfNorm(velocity, exact_velocity, T) << std::endl;
  }

  return 0;
}
