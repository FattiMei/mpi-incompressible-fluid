#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Real.h"
#include <cstddef>

namespace mif {

// A class containing all constants needed for execution and derived constants.
class Constants {
public:
  // Core constants.
  const Real x_size;
  const Real y_size;
  const Real z_size;
  const size_t Nx;
  const size_t Ny;
  const size_t Nz;
  const Real Re;
  const Real final_time;
  const unsigned int num_time_steps;
    Real max_velocity;

  // Derived constants (computed here once for efficiency).
  const Real dt;
  const Real dx;
  const Real dy;
  const Real dz;
  const size_t row_size;
  const size_t matrix_size;
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

  // Constructor.
  Constants(size_t Nx, size_t Ny, size_t Nz, Real x_size, Real y_size,
            Real z_size, Real Re, Real final_time, unsigned int num_time_steps) noexcept;
};

} // namespace mif

#endif