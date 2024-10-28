#include "Manufactured.h"
#include "Norms.h"
#include "Timestep.h"
#include <iostream>

double Reynolds = 1e6;
constexpr Real Lx = 1.0;
constexpr Real Ly = 1.0;
constexpr Real Lz = 1.0;
constexpr Real T = 1e-4;

using namespace mif;

int main() {
  std::cout << "deltax,deltat,l1,l2,linf" << std::endl;

  for (int N = 32, ntime_steps = 1; N < 256; N *= 2, ntime_steps *= 2) {
    const size_t Nx = N + 1;
    const size_t Ny = Nx;
    const size_t Nz = Nx;

    const Constants constants(Nx, Ny, Nz, Lx, Ly, Lz, Reynolds, T, ntime_steps);

    VelocityTensor velocity(constants);
    VelocityTensor velocity_buffer(constants);
    VelocityTensor rhs_buffer(constants);

    TimeVectorFunction exact_velocity(u_exact, v_exact, w_exact);
    velocity.set(exact_velocity.set_time(0.0), true);

    TimeVectorFunction forcing_term(forcing_x, forcing_y, forcing_z);

    for (int step = 0; step < ntime_steps; ++step) {
      const Real t = step * constants.dt;

      timestep(velocity, velocity_buffer, rhs_buffer, t);
    }

    std::cout << constants.dx << ',' << constants.dt << ','
              << ErrorL1Norm(velocity, T) << ',' << ErrorL2Norm(velocity, T)
              << ',' << ErrorLInfNorm(velocity, T) << std::endl;
  }

  return 0;
}
