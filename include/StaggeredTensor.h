
#ifndef STAGGERED_TENSOR_H
#define STAGGERED_TENSOR_H

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#pragma GCC diagnostic pop 
#include "Constants.h"
#include "Tensor.h"

namespace mif {

enum StaggeringDirection {
  x, y, z, none
};

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
  StaggeredTensor(const Constants &constants, const StaggeringDirection &staggering);

  // Send data to neighbouring processors using MPI.
  // This will use the tags in [base_tag, base_tag+3].
  // This includes periodic BC across processor boundaries.
  void send_mpi_data(int base_tag);

  // Receive data from neighbouring processors using MPI.
  // This will use the tags in [base_tag, base_tag+3].
  void receive_mpi_data(int base_tag);

  // Apply periodic BC on the boundaries specified in Constants.
  // This does not apply periodic BC across processor boundaries.
  void apply_periodic_bc();

  // Swapping data with another tensor by flipping the buffer pointers is not
  // enough, as the MPI addressing will be messed up. This function should be
  // used to recompute the MPI addressing after a swap. Particularly, it
  // updates the addresses used to send and receive data to and from
  // neighbouring processors.
  void recompute_mpi_addressing();

  const Constants &constants;

  // Direction in which the tensor is staggered.
  StaggeringDirection staggering;

  // A MPI datatype representing a slice with constant y coordinate.
  MPI_Datatype Slice_type_constant_y; 

  // A MPI datatype representing a slice with constant z coordinate.
  MPI_Datatype Slice_type_constant_z;
  
  // These eight addresses are used to send and receive data to and from
  // neighbouring processors. They are computed in recompute_mpi_addressing.
  // The data that a given processor sends to its neighbours is stored in
  // the very first and last available cells in a given direction.
  // The data that a given processor receives from its neighbours is stored in
  // the second and second-to-last available cells in a given direction.
  void *min_addr_recv_y;
  void *max_addr_recv_y;
  void *min_addr_recv_z;
  void *max_addr_recv_z;
  void *min_addr_send_y;
  void *max_addr_send_y;
  void *min_addr_send_z;
  void *max_addr_send_z;
  
  // These tensors store a slice of the overall tensor with a constant y value,
  // i.e. the y value to send or receive to previous and next processors respectively.
  Tensor<Real, 2U, size_t> prev_y_slice_recv;
  Tensor<Real, 2U, size_t> next_y_slice_recv;
  Tensor<Real, 2U, size_t> prev_y_slice_send;
  Tensor<Real, 2U, size_t> next_y_slice_send;

  /*!
   * Evaluate the function f, depending on t,x,y,z, on an index of this
   * tensor.
   *
   * @param i The index for the x direction.
   * @param j The index for the y direction.
   * @param k The index for the z direction.
   * @param f The function to evaluate.
   * @param constants An object containing information on the domain.
   */
  inline Real evaluate_function_at_index_unstaggered(
    Real time, int i, int j, int k,
      const std::function<Real(Real, Real, Real, Real)> &f) const {
    return f(time, constants.min_x_global + constants.dx * (constants.base_i+i), 
             constants.min_y_global + constants.dy * (constants.base_j+j),
             constants.min_z_global + constants.dz * (constants.base_k+k));    
  }

  // Do the same, but without the time dependency.
  inline Real evaluate_function_at_index_unstaggered(
    int i, int j, int k,
      const std::function<Real(Real, Real, Real)> &f) const {
    return f(constants.min_x_global + constants.dx * (constants.base_i+i), 
             constants.min_y_global + constants.dy * (constants.base_j+j),
             constants.min_z_global + constants.dz * (constants.base_k+k)); 
  }

  // Do the same considering staggering.
  virtual inline Real evaluate_function_at_index(
    Real time, int i, int j, int k,
    const std::function<Real(Real, Real, Real, Real)> &f) const {
      return evaluate_function_at_index_unstaggered(time, i, j, k, f);
  }

  // Do the same considering staggering.
  virtual inline Real evaluate_function_at_index(
      int i, int j, int k,
      const std::function<Real(Real, Real, Real)> &f) const {
      return evaluate_function_at_index_unstaggered(i, j, k, f);
  }

  // A debug function to print the tensor.
  void print() const;
  void print(const std::function<bool(Real)> &filter) const;
  void print_inline() const;

  // Set the values of this tensor to the corresponding values of f, evaluated at all points.
  void set(const std::function<Real(Real, Real, Real)> &f, bool include_border);
};

} // mif

#endif // STAGGERED_TENSOR_H