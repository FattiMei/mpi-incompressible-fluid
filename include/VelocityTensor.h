#ifndef VELOCITY_TENSOR_H
#define VELOCITY_TENSOR_H

#include "StaggeredTensor.h"
#include "VectorFunction.h"

namespace mif {

// Tensor staggered in the x direction.
class UTensor : public StaggeredTensor {
public:
  UTensor(const Constants &constants)
      : StaggeredTensor(constants, StaggeringDirection::x) {}
  UTensor(const UTensor&) = delete;

  inline Real evaluate_function_at_index(
      int i, int j, int k,
      const std::function<Real(Real, Real, Real)> &f) const override {
    return f(constants.min_x_global + constants.dx * (constants.base_i+i) - constants.dx_over_2, 
             constants.min_y_global + constants.dy * (constants.base_j+j),
             constants.min_z_global + constants.dz * (constants.base_k+k)); 
  }

  inline Real evaluate_function_at_index(
      Real time, int i, int j, int k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.min_x_global + constants.dx * (constants.base_i+i) - constants.dx_over_2, 
             constants.min_y_global + constants.dy * (constants.base_j+j),
             constants.min_z_global + constants.dz * (constants.base_k+k)); 
  }
};

// Tensor staggered in the y direction.
class VTensor : public StaggeredTensor {
public:
  VTensor(const Constants &constants)
      : StaggeredTensor(constants, StaggeringDirection::y) {}
  VTensor(const VTensor&) = delete;

  inline Real evaluate_function_at_index(
      int i, int j, int k,
      const std::function<Real(Real, Real, Real)> &f) const override {
    return f(constants.min_x_global + constants.dx * (constants.base_i+i), 
             constants.min_y_global + constants.dy * (constants.base_j+j) - constants.dy_over_2,
             constants.min_z_global + constants.dz * (constants.base_k+k)); 
  }

  inline Real evaluate_function_at_index(
      Real time, int i, int j, int k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.min_x_global + constants.dx * (constants.base_i+i), 
             constants.min_y_global + constants.dy * (constants.base_j+j) - constants.dy_over_2,
             constants.min_z_global + constants.dz * (constants.base_k+k)); 
  }
};

// Tensor staggered in the z direction.
class WTensor : public StaggeredTensor {
public:
  WTensor(const Constants &constants)
      : StaggeredTensor(constants, StaggeringDirection::z) {}
  WTensor(const WTensor&) = delete;

  inline Real evaluate_function_at_index(
      int i, int j, int k,
      const std::function<Real(Real, Real, Real)> &f) const override {
    return f(constants.min_x_global + constants.dx * (constants.base_i+i), 
             constants.min_y_global + constants.dy * (constants.base_j+j),
             constants.min_z_global + constants.dz * (constants.base_k+k) - constants.dz_over_2); 
  }

  inline Real evaluate_function_at_index(
      Real time, int i, int j, int k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.min_x_global + constants.dx * (constants.base_i+i), 
             constants.min_y_global + constants.dy * (constants.base_j+j),
             constants.min_z_global + constants.dz * (constants.base_k+k) - constants.dz_over_2); 
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
  const Constants &constants;

  VelocityTensor(const Constants &constants);
  VelocityTensor(const VelocityTensor&) = delete;

  // Swap this tensor's data with another's in constant time by swapping
  // pointers.
  void swap_data(VelocityTensor &other);

  // Set all components of the tensor in all points using the respective
  // components of the function.
  void set(const VectorFunction &f, bool include_border);

  // Apply boundary conditions to all components of the velocity
  // on all Dirichlet boundaries. Dirichlet BC are used on boundaries for which
  // constants does not specify periodic BC. Periodic BC are used
  // elsewhere.
  // The function assumes the velocity field is divergence free.
  void apply_bc(const VectorFunction &exact_velocity);
};

} // namespace mif

#endif // VELOCITY_TENSOR_H