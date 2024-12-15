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
Constants::Constants(size_t Nx_domains, size_t Ny_domains_global, size_t Nz_domains_global, 
                     Real x_size, Real y_size_global, Real z_size_global, Real Re, 
                     Real final_time, unsigned int num_time_steps,
                     int Py, int Pz, int rank)
    : Nx_domains(Nx_domains), Ny_domains_global(Ny_domains_global), Nz_domains_global(Nz_domains_global),
      x_size(x_size), y_size_global(y_size_global), z_size_global(z_size_global), 
      Re(Re), final_time(final_time), num_time_steps(num_time_steps),
      Py(Py), Pz(Pz), rank(rank), y_rank(rank / Pz), z_rank(rank % Pz),
      dt(final_time / num_time_steps), dx(x_size / Nx_domains), 
      dy(y_size_global / Ny_domains_global), dz(z_size_global / Nz_domains_global),
      one_over_2_dx(1 / (2 * dx)), one_over_2_dy(1 / (2 * dy)), one_over_2_dz(1 / (2 * dz)), 
      one_over_8_dx(1 / (8 * dx)), one_over_8_dy(1 / (8 * dy)), 
      one_over_8_dz(1 / (8 * dz)), one_over_dx2_Re(1 / (Re * dx * dx)), 
      one_over_dy2_Re(1 / (Re * dy * dy)), one_over_dz2_Re(1 / (Re * dz * dz)), 
      dx_over_2(dx / 2), dy_over_2(dy / 2), dz_over_2(dz / 2), one_over_dx(1 / dx), 
      one_over_dy(1 / dy), one_over_dz(1 / dz),
      P(Py * Pz), Ny_domains_local(Ny_domains_global / Py), Nz_domains_local(Nz_domains_global / Pz), 
      Nx_staggered(Nx_domains + 2UL), Ny_staggered(Ny_domains_local + 2UL), Nz_staggered(Nz_domains_local + 2UL), 
      Nx(Nx_staggered - 1UL), Ny((y_rank == (Py - 1)) ? Ny_staggered - 1UL : Ny_staggered), Nz((z_rank == (Pz - 1)) ? Nz_staggered - 1UL : Nz_staggered),
      
      // Calculate the size of the domain for the current processor.
      y_size_local(y_size_global / Py), z_size_local(z_size_global / Pz), 
      
      // Calculate the minimum and maximum values of y and z for the current
      // processor. Each processor will have a different subset of the domain.
      min_y(y_size_local * y_rank), max_y(min_y + y_size_local + ((y_rank == Py - 1) ? 0.0 : dy)),
      min_z(z_size_local * z_rank), max_z(min_z + z_size_local + ((z_rank == Pz - 1) ? 0.0 : dz)),
      
      // Calculate the neighbouring processors in the y and z directions based
      // on the rank.
      prev_proc_y((y_rank == 0)? -1: rank-Pz), next_proc_y((y_rank == Py - 1)? -1: rank+Pz),
      prev_proc_z((z_rank == 0)? -1: rank-1), next_proc_z((z_rank == Pz - 1)? -1: rank+1) {

  // Ensure that the number of domains in each direction is positive.
  assert(Nx > 0 && Ny > 0 && Nz > 0);

  // Ensure that the number of time steps and final time are positive.
  assert(num_time_steps > 0 && final_time > 0);

  // Ensure that the number of processors in each direction is positive.
  assert(Py > 0 && Pz > 0);
  
  // Ensure that the local domains multiplied by the number of processors equals the global domains.
  assert(Ny_domains_local * Py == Ny_domains_global);
  assert(Nz_domains_local * Pz == Nz_domains_global);

  // Ensure that the rank is within the valid range.
  assert(y_rank >= 0 && y_rank < Py);
  assert(z_rank >= 0 && z_rank < Pz);
  assert(rank >= 0 && rank < P);

  // Ensure that the previous and next processors in y direction are valid.
  assert(prev_proc_y >= -1 && prev_proc_y < rank && (next_proc_y == -1 || next_proc_y > rank) && next_proc_y < P);
  assert(prev_proc_z >= -1 && prev_proc_z < rank && (next_proc_z == -1 || next_proc_z > rank) && next_proc_z < P);
}

} // namespace mif