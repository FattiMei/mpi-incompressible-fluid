#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Real.h"
#include <cstddef>

namespace mif {

// A class containing all constants needed for execution and derived constants.
class Constants {
public:
  // Core constants.
  const size_t Nx_domains_global;
  const size_t Ny_domains_global;
  const size_t Nz_domains;
  const Real x_size_global;
  const Real y_size_global;
  const Real z_size;
  const Real Re;
  const Real final_time;
  const unsigned int num_time_steps;
  const int Px;
  const int Py; 
  const int x_rank;
  const int y_rank;

  // Derived constants (computed here once for efficiency).
  const Real dt;
  const Real dx;
  const Real dy;
  const Real dz;
  const Real one_over_2_dx;
  const Real one_over_2_dy;
  const Real one_over_2_dz;
  const Real one_over_8_dx;
  const Real one_over_8_dy;
  const Real one_over_8_dz;
  const Real one_over_dx2_Re;
  const Real one_over_dy2_Re;
  const Real one_over_dz2_Re;
  const Real dx_over_2;
  const Real dy_over_2;
  const Real dz_over_2;
  const Real one_over_dx;
  const Real one_over_dy;
  const Real one_over_dz;
  const int P;
  const size_t Nx_domains_local;
  const size_t Ny_domains_local;
  const size_t Nx_staggered;
  const size_t Ny_staggered;
  const size_t Nz_staggered;
  const size_t Nx;
  const size_t Ny;
  const size_t Nz;
  const Real x_size_local;
  const Real y_size_local;
  const Real min_x;
  const Real max_x;
  const Real min_y;
  const Real max_y;

  // Constructor.
  Constants(size_t Nx_domains_global, size_t Ny_domains_global, size_t Nz_domains, 
            Real x_size, Real y_size, Real z_size, Real Re, Real final_time, unsigned int num_time_steps,
            int Px, int Py, int x_rank, int y_rank);
};

} // namespace mif

#endif