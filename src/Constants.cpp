#include "Constants.h"
#include <cassert>

#include <iostream>

namespace mif {

Constants::Constants(size_t Nx_domains_global, size_t Ny_domains_global, size_t Nz_domains, 
                     Real x_size_global, Real y_size_global, Real z_size, Real Re, 
                     Real final_time, unsigned int num_time_steps,
                     int Px, int Py, int x_rank, int y_rank)
    : Nx_domains_global(Nx_domains_global), Ny_domains_global(Ny_domains_global), Nz_domains(Nz_domains),
      x_size_global(x_size_global), y_size_global(y_size_global), z_size(z_size), 
      Re(Re), final_time(final_time), num_time_steps(num_time_steps),
      Px(Px), Py(Py), x_rank(x_rank), y_rank(y_rank),
      dt(final_time / num_time_steps), dx(x_size_global / Nx_domains_global), 
      dy(y_size_global / Ny_domains_global), dz(z_size / Nz_domains),
      one_over_2_dx(1 / (2 * dx)), one_over_2_dy(1 / (2 * dy)), one_over_2_dz(1 / (2 * dz)), 
      one_over_8_dx(1 / (8 * dx)), one_over_8_dy(1 / (8 * dy)), 
      one_over_8_dz(1 / (8 * dz)), one_over_dx2_Re(1 / (Re * dx * dx)), 
      one_over_dy2_Re(1 / (Re * dy * dy)), one_over_dz2_Re(1 / (Re * dz * dz)), 
      dx_over_2(dx / 2), dy_over_2(dy / 2), dz_over_2(dz / 2), one_over_dx(1 / dx), 
      one_over_dy(1 / dy), one_over_dz(1 / dz),
      P(Px * Py), Nx_domains_local(Nx_domains_global / Px), Ny_domains_local(Ny_domains_global / Py), 
      Nx_staggered(Nx_domains_local + 2UL), Ny_staggered(Ny_domains_local + 2UL), Nz_staggered(Nz_domains + 2UL), 
      Nx(x_rank == Px - 1 ? Nx_staggered - 1UL : Nx_staggered), Ny(y_rank == Py - 1 ? Ny_staggered - 1UL : Ny_staggered), Nz(Nz_staggered - 1UL),
      x_size_local(x_size_global / Px), y_size_local(y_size_global / Py),
      min_x(x_size_local * x_rank), max_x(min_x + x_size_local + (x_rank == Px - 1) ? 0.0 : dx),
      min_y(y_size_local * y_rank), max_y(min_y + y_size_local + (y_rank == Py - 1) ? 0.0 : dy) {
  assert(Nx > 0 && Ny > 0 && Nz > 0);
  assert(num_time_steps > 0 && final_time > 0);
  assert(Px > 0 && Py > 0);
  assert(Nx_domains_local * Px == Nx_domains_global);
  assert(Ny_domains_local * Py == Ny_domains_global);
}

} // namespace mif