#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Real.h"
#include <array>
#include <cstddef>

namespace mif {

// A class containing all constants needed for execution and derived constants.
class Constants {
public:
  // Core constants.
  // In the case of periodic BC, Nx_global, Ny_global and Nz_global
  // include the copy of the first point at the end of the domain,
  // but not the copy of the second point after the end.
  const size_t Nx_global;
  const size_t Ny_global;
  const size_t Nz_global;
  const Real x_size;
  const Real y_size_global;
  const Real z_size_global;
  const Real min_x_global;
  const Real min_y_global;
  const Real min_z_global;
  const Real Re;
  const Real final_time;
  const unsigned int num_time_steps;
  const std::array<bool, 3> periodic_bc;

  // MPI constants.
  const int Py; 
  const int Pz;
  const int rank;
  const int y_rank;
  const int z_rank;

  // Derived constants (computed here once for efficiency).
  const Real dt;
  const size_t Nx_domains;
  const size_t Ny_domains_global;
  const size_t Nz_domains_global;
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

  // Derived constants for MPI and the staggered grid.
  // "owner" values ignore periodic BC.
  // Note: base_i, j and k may be negative due to periodic BC.
  const int P;
  const size_t Ny_owner;
  const size_t Nz_owner;
  const size_t Nx;
  const size_t Ny;
  const size_t Nz;
  const size_t Nx_staggered;
  const size_t Ny_staggered;
  const size_t Nz_staggered;
  const int base_i;
  const int base_j;
  const int base_k;

  // Derived constants for the MPI communication.
  // The neighbors include those related to periodic boundaries.
  const int prev_proc_y;
  const int next_proc_y;
  const int prev_proc_z;
  const int next_proc_z;

  // Constructor.
  Constants(size_t Nx_global, size_t Ny_global, size_t Nz_global, 
            Real x_size, Real y_size_global, Real z_size_global, 
            Real min_x_global, Real min_y_global, Real min_z_global, 
            Real Re, Real final_time, unsigned int num_time_steps,
            int Py, int Pz, int rank, const std::array<bool, 3> &periodic_bc);
};

} // namespace mif

#endif