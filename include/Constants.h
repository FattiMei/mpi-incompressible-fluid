#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Real.h"
#include <cstddef>

namespace mif {

// A class containing all constants needed for execution and derived constants.
class Constants {
public:
  // Core constants.
  const size_t Nx_domains;
  const size_t Ny_domains_global;
  const size_t Nz_domains_global;
  const Real x_size;
  const Real y_size_global;
  const Real z_size_global;
  const Real Re;
  const Real final_time;
  const unsigned int num_time_steps;

  // MPI constants.
  const int Py; 
  const int Pz;
  const int rank;
  const int y_rank;
  const int z_rank;

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

  // Derived constants for the staggered grid.
  const int P;
  const size_t Ny_domains_local;
  const size_t Nz_domains_local;
  const size_t Nx_staggered;
  const size_t Ny_staggered;
  const size_t Nz_staggered;
  const size_t Nx;
  const size_t Ny;
  const size_t Nz;

  // Derived constants for the local domain.
  const Real y_size_local;
  const Real z_size_local;
  const Real min_y;
  const Real max_y;
  const Real min_z;
  const Real max_z;

  // Derived constants for the MPI communication.
  const int prev_proc_y;
  const int next_proc_y;
  const int prev_proc_z;
  const int next_proc_z;

  // Constructor.
  Constants(size_t Nx_domains, size_t Ny_domains_global, size_t Nz_domains_global, 
            Real x_size, Real y_size, Real z_size, Real Re, Real final_time, unsigned int num_time_steps,
            int Py, int Pz, int rank);
};

} // namespace mif

#endif