#ifndef VELOCITY_TENSOR_H
#define VELOCITY_TENSOR_H

#include <mpi.h>
#include "Constants.h"
#include "Tensor.h"
#include "VectorFunction.h"

namespace mif {

/*!
 * @class StaggeredTensor
 * @brief A tensor with staggered components.
 *
 * This class represents a tensor with staggered components, i.e., components
 * that are offset by half a grid cell in one of the directions. We further
 * inherit from this class to create tensors staggered in the x, y, and z
 * directions, see UTensor, VTensor, and WTensor.
 *
 * @param constants An object containing information on the domain.
 * @param in_dimensions The dimensions of the tensor.
 */
class StaggeredTensor : public Tensor<Real, 3U, size_t> {
public:
  StaggeredTensor(const std::array<size_t, 3> &in_dimensions, const Constants &constants);

  // Send data to neighbouring processors using MPI.
  // This will use the tags in [base_tag, base_tag+3].
  void send_mpi_data(int base_tag);

  // Swapping data with another tensor by flipping the buffer pointers is not
  // enough, as the MPI addressing will be messed up. This function should be
  // used to recompute the MPI addressing after a swap. Particularly, it
  // updates the addresses used to send and receive data to and from
  // neighbouring processors.
  void recompute_mpi_addressing();

  const Constants &constants;
  // A MPI datatype representing a slice with constant x coordinate.
  MPI_Datatype Constant_slice_type_x;

  // A MPI datatype representing a slice with constant y coordinate.
  MPI_Datatype Constant_slice_type_y; 
  
  // These eight addresses are used to send and receive data to and from
  // neighbouring processors. They are computed in recompute_mpi_addressing.
  // The data that a given processor sends to its neighbours is stored in
  // the very first and last available cells in a given direction.
  // The data that a given processor receives from its neighbours is stored in
  // the second and second-to-last available cells in a given direction.
  void *min_addr_recv_x;
  void *max_addr_recv_x;
  void *min_addr_recv_y;
  void *max_addr_recv_y;
  void *min_addr_send_x;
  void *max_addr_send_x;
  void *min_addr_send_y;
  void *max_addr_send_y;

  /*!
   * Evaluate the function f, depending on x,y,z, on an index of this
   * tensor.
   *
   * @param i The index for the x direction.
   * @param j The index for the y direction.
   * @param k The index for the z direction.
   * @param f The function to evaluate.
   * @param constants An object containing information on the domain.
   */
  virtual inline Real evaluate_function_at_index(
      size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real)> &f) const = 0;

  // Do the same, but for a function depending on time as well, as the
  // first input.
  virtual inline Real evaluate_function_at_index(
      Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const = 0;
  
  // Do the same, but without considering staggering.
  inline Real evaluate_function_at_index_unstaggered(
    size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real)> &f) const {
    return f(constants.min_x + constants.dx * i, constants.min_y + constants.dy * j,
             constants.dz * k);
  }

  // Do the same, but without considering staggering, for a function
  // depending on time as well.
  inline Real evaluate_function_at_index_unstaggered(
    Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const {
    return f(time, constants.min_x + constants.dx * i, constants.min_y + constants.dy * j,
             constants.dz * k);    
  }

  // A debug function to print the tensor.
  void print() const;
  void print(const std::function<bool(Real)> &filter) const;
};

// Tensor staggered in the x direction.
class UTensor : public StaggeredTensor {
public:
  UTensor(const Constants &constants)
      : StaggeredTensor({constants.Nx_staggered, constants.Ny, constants.Nz}, constants) {}

  inline Real evaluate_function_at_index(
      size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real)> &f) const override {
    return f(constants.min_x + constants.dx * i - constants.dx_over_2, constants.min_y + constants.dy * j,
             constants.dz * k);
  }

  inline Real evaluate_function_at_index(
      Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.min_x + constants.dx * i - constants.dx_over_2, constants.min_y + constants.dy * j,
             constants.dz * k);
  }
};

// Tensor staggered in the y direction.
class VTensor : public StaggeredTensor {
public:
  VTensor(const Constants &constants)
      : StaggeredTensor({constants.Nx, constants.Ny_staggered, constants.Nz}, constants) {}

  inline Real evaluate_function_at_index(
      size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real)> &f) const override {
    return f(constants.min_x + constants.dx * i, constants.min_y + constants.dy * j - constants.dy_over_2,
             constants.dz * k);
  }

  inline Real evaluate_function_at_index(
      Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.min_x + constants.dx * i, constants.min_y + constants.dy * j - constants.dy_over_2,
             constants.dz * k);
  }
};

// Tensor staggered in the z direction.
class WTensor : public StaggeredTensor {
public:
  WTensor(const Constants &constants)
      : StaggeredTensor({constants.Nx, constants.Ny, constants.Nz_staggered}, constants) {}

  inline Real evaluate_function_at_index(
      size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real)> &f) const override {
    return f(constants.min_x + constants.dx * i, constants.min_y + constants.dy * j,
             constants.dz * k - constants.dz_over_2);
  }

  inline Real evaluate_function_at_index(
      Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.min_x + constants.dx * i, constants.min_y + constants.dy * j,
             constants.dz * k - constants.dz_over_2);
  }
};

/*!
 * @class VelocityTensor
 * @brief A collection of 3 tensors representing the 3 velocity components.
 *
 * This class is mainly used to abstract the velocity field into a single
 * object, which can be easily manipulated and passed around.
 */
class VelocityTensor {
public:
  UTensor u;
  VTensor v;
  WTensor w;
  std::array<StaggeredTensor *, 3> components;
  const Constants constants;

  VelocityTensor(const Constants &constants);

  // Swap this tensor's data with another's in constant time by swapping
  // pointers.
  void swap_data(VelocityTensor &other);

  // Set all components of the tensor in all points using the respective
  // components of the function.
  void set(const VectorFunction &f, bool include_border);

  // Same as set, but meant for setting intial conditions.
  void set_initial(const VectorFunction &f);

  // Apply Dirichlet boundary conditions to all components of the velocity
  // on all boundaries. The function assumes the velocity field is
  // divergence free.
  void apply_all_dirichlet_bc(Real time);
};

} // namespace mif

#endif // VELOCITY_TENSOR_H