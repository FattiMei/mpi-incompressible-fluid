#include "Constants.h"
#include <cassert>

#include <iostream>

namespace mif {

/**
 * @brief Constructor for the Constants class.
 * 
 * This constructor initializes all the constants needed for execution, and
 * derived constants. Note that these constants are relative to the current
 * processor, since we are using a domain decomposition. This means that these
 * will be different for each processor, and not within the entire domain.
 * 
 * Each processor in MPI has a rank (its ID). This rank is decomposed into
 * x_rank and y_rank, by using the modulo and division operators with respect to
 * the number of processors in the x direction (Px). This decomposition makes it
 * easier to determine the position of the processor in the grid of processors.
 * 
 * Nx, Ny and Nz represent the number of points in the x, y, and z directions
 * for the current processor, respectively. They take into account the presence
 * of ghost points, which are used for communication between processors.
 * Normally, these would coincide with the staggered grid points, except for the
 * last processor in each direction, which has one less point in the x and y
 * directions.
 * 
 * The staggered values of Nx, Ny, and Nz are calculated by adding 2 to the
 * local values, to account for the ghost points. Note that the local values are
 * the trivial division of the global values by the number of processors in each
 * direction.
 * 
 * These values are used to produce the minimum and maximum values of x and y
 * for the current processor, `min_x`, `max_x`, `min_y`, and `max_y`.
 * 
 * The previous and next processors in the x and y directions are also
 * calculated, to facilitate the implementation of the MPI communication. These
 * are calculated by checking the x and y ranks of the current processor,
 * explained above.
 * 
 * The z direction is not decomposed, as it is not parallelized. The number of
 * domains in the z direction is the same for all processors, and the number of
 * points in the z direction is the same for all processors.
 * 
 * @param Nx_domains_global Number of domains in the x direction globally.
 * @param Ny_domains_global Number of domains in the y direction globally.
 * @param Nz_domains Number of domains in the z direction.
 * @param x_size_global Size of the domain in the x direction globally.
 * @param y_size_global Size of the domain in the y direction globally.
 * @param z_size Size of the domain in the z direction.
 * @param Re Reynolds number.
 * @param final_time End of the simulation.
 * @param num_time_steps Number of time steps for the simulation.
 * @param Px Number of processors in the x direction.
 * @param Py Number of processors in the y direction.
 * @param rank Rank of the current processor.
 */
Constants::Constants(size_t Nx_global, size_t Ny_global, size_t Nz_global, 
                     Real x_size, Real y_size_global, Real z_size_global, 
                     Real min_x_global, Real min_y_global, Real min_z_global,
                     Real Re, Real final_time, unsigned int num_time_steps,
                     int Py, int Pz, int rank, const std::array<bool, 3> &periodic_bc)
    : Nx_global(Nx_global), Ny_global(Ny_global), Nz_global(Nz_global),
      x_size(x_size), y_size_global(y_size_global), z_size_global(z_size_global), 
      min_x_global(min_x_global), min_y_global(min_y_global), min_z_global(min_z_global),
      Re(Re), final_time(final_time), num_time_steps(num_time_steps),
      periodic_bc(periodic_bc), 
      Py(Py), Pz(Pz), rank(rank), y_rank(rank / Pz), z_rank(rank % Pz),
      dt(final_time / num_time_steps),
      Nx_domains(Nx_global-1), Ny_domains_global(Ny_global-1), Nz_domains_global(Nz_global-1),
      dx(x_size / Nx_domains), dy(y_size_global / Ny_domains_global), dz(z_size_global / Nz_domains_global),
      one_over_2_dx(1 / (2 * dx)), one_over_2_dy(1 / (2 * dy)), one_over_2_dz(1 / (2 * dz)), 
      one_over_8_dx(1 / (8 * dx)), one_over_8_dy(1 / (8 * dy)), 
      one_over_8_dz(1 / (8 * dz)), one_over_dx2_Re(1 / (Re * dx * dx)), 
      one_over_dy2_Re(1 / (Re * dy * dy)), one_over_dz2_Re(1 / (Re * dz * dz)), 
      dx_over_2(dx / 2), dy_over_2(dy / 2), dz_over_2(dz / 2), one_over_dx(1 / dx), 
      one_over_dy(1 / dy), one_over_dz(1 / dz), P(Py * Pz), 
      Ny_owner((Ny_global - periodic_bc[1]) / Py + ((static_cast<size_t>(y_rank) < ((Ny_global - periodic_bc[1]) % Py)) ? 1 : 0)),
      Nz_owner((Nz_global - periodic_bc[2]) / Pz + ((static_cast<size_t>(z_rank) < ((Nz_global - periodic_bc[2]) % Pz)) ? 1 : 0)),
      Nx((periodic_bc[0] ? Nx_global+1 : Nx_global)),
      Ny((Py == 1) ? (periodic_bc[1] ? Ny_global+1 : Ny_global) : (((y_rank == (Py-1) && !periodic_bc[1]) || (y_rank == 0 && !periodic_bc[1])) ? Ny_owner+1 : Ny_owner+2)),
      Nz((Pz == 1) ? (periodic_bc[2] ? Nz_global+1 : Nz_global) : (((z_rank == (Pz-1) && !periodic_bc[2]) || (z_rank == 0 && !periodic_bc[2])) ? Nz_owner+1 : Nz_owner+2)),
      Nx_staggered(periodic_bc[0] ? Nx : Nx + 1), 
      Ny_staggered((y_rank == (Py-1)) ? Ny + 1 : Ny), 
      Nz_staggered((z_rank == (Pz-1)) ? Nz + 1 : Nz),

      // Each processor will have a different subset of the domain.
      // If the domain cannot be split evenly, bonus points go to the first processors
      // in each direction.
      // Due to periodic BC, there may be a bonus point before and after the end of 
      // the domain.
      base_i(periodic_bc[0] ? -1 : 0),
      base_j((Ny_global - periodic_bc[1]) / Py * y_rank + std::min(static_cast<size_t>(y_rank), (Ny_global - periodic_bc[1]) % Py) - ((y_rank > 0 || periodic_bc[1]) ? 1: 0)),
      base_k((Nz_global - periodic_bc[2]) / Pz * z_rank + std::min(static_cast<size_t>(z_rank), (Nz_global - periodic_bc[2]) % Pz) - ((z_rank > 0 || periodic_bc[2]) ? 1: 0)),
      
      // Calculate the neighbouring processors in the y and z directions based
      // on the rank.
      prev_proc_y((y_rank == 0)? ((Py > 1 && periodic_bc[1]) ? rank+(Py-1)*Pz: -1): rank-Pz), 
      next_proc_y((y_rank == Py - 1)? ((Py > 1 && periodic_bc[1]) ? rank-(Py-1)*Pz: -1): rank+Pz),
      prev_proc_z((z_rank == 0)? ((Pz > 1 && periodic_bc[2]) ? rank+Pz-1: -1): rank-1), 
      next_proc_z((z_rank == Pz - 1)? ((Pz > 1 && periodic_bc[2]) ? rank-(Pz-1): -1): rank+1) {

  // Ensure that the number of domains in each direction is at least 2.
  assert(Nx >= 2 && Ny >= 2 && Nz >= 2);

  // Ensure that the number of time steps and final time are positive.
  assert(num_time_steps > 0 && final_time > 0);

  // Ensure that the number of processors in each direction is positive.
  assert(Py > 0 && Pz > 0);

  // Ensure that the rank is within the valid range.
  assert(y_rank >= 0 && y_rank < Py);
  assert(z_rank >= 0 && z_rank < Pz);
  assert(rank >= 0 && rank < P);

  // Ensure that the previous and next processors in y direction are valid.
  assert(prev_proc_y >= -1 && prev_proc_y < P && next_proc_y >= -1 && next_proc_y < P);
  assert(prev_proc_z >= -1 && prev_proc_z < P && next_proc_z >= -1 && next_proc_z < P);
}

} // namespace mif