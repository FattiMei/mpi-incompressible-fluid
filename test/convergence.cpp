#include "Manufactured.h"
#include "Norms.h"
#include "Timestep.h"
#include <iostream>

Real Reynolds = 4700;
constexpr Real Lx = 1.0;
constexpr Real Ly = 1.0;
constexpr Real Lz = 1.0;
constexpr Real T = 1;

using namespace mif;

int main() {
  std::cout << "deltax,deltat,l1,l2,linf" << std::endl;
const Real target_cfl = 0.2;
  for (int N = 4, ntime_steps = 64; N < 64;  N *= 2, ntime_steps *= 2) {
    const size_t Nx = N + 1;
    const size_t Ny = Nx;
    const size_t Nz = Nx;


    const Constants constants(Nx, Ny, Nz, Lx, Ly, Lz, (Reynolds), T, ntime_steps);

    VelocityTensor velocity(constants);
    VelocityTensor velocity_buffer(constants);
    VelocityTensor rhs_buffer(constants);

    TimeVectorFunction exact_velocity(u_exact, v_exact, w_exact);
    velocity.set(exact_velocity, true,0);

    TimeVectorFunction forcing_term(forcing_x, forcing_y, forcing_z);
    Real t = 0.0;
    size_t step = 0;
    Real last_dt = constants.dt;
    while (t < T) {
      //prnt t every 100 iterations
      last_dt=timestep(velocity, velocity_buffer, rhs_buffer, t, target_cfl, last_dt);
      t += last_dt;
      step++;
    }

    std::cout << constants.dx << ',' << t/step << ','
              << ErrorL1Norm(velocity, t) << ',' << ErrorL2Norm(velocity, t)
              << ',' << ErrorLInfNorm(velocity, t) << std::endl;
  }

  return 0;
}
