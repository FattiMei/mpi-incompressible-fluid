#ifndef VELOCITY_TENSOR_H
#define VELOCITY_TENSOR_H

#include "StaggeredTensor.h"
#include "VectorFunction.h"

namespace mif {

// Tensor staggered in the x direction.
class UTensor : public StaggeredTensor {
public:
  UTensor(const Constants &constants)
      : StaggeredTensor({constants.Nx_staggered, constants.Ny, constants.Nz}, constants) {}

  inline Real evaluate_function_at_index(
      size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real)> &f) const override {
    return f(constants.dx * i - constants.dx_over_2, constants.min_y + constants.dy * j,
             constants.min_z + constants.dz * k);
  }

  inline Real evaluate_function_at_index(
      Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.dx * i - constants.dx_over_2, constants.min_y + constants.dy * j,
             constants.min_z + constants.dz * k);
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
    return f(constants.dx * i, constants.min_y + constants.dy * j - constants.dy_over_2,
             constants.min_z + constants.dz * k);
  }

  inline Real evaluate_function_at_index(
      Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.dx * i, constants.min_y + constants.dy * j - constants.dy_over_2,
             constants.min_z + constants.dz * k);
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
    return f(constants.dx * i, constants.min_y + constants.dy * j,
             constants.min_z + constants.dz * k - constants.dz_over_2);
  }

  inline Real evaluate_function_at_index(
      Real time, size_t i, size_t j, size_t k,
      const std::function<Real(Real, Real, Real, Real)> &f) const override {
    return f(time, constants.dx * i, constants.min_y + constants.dy * j,
             constants.min_z + constants.dz * k - constants.dz_over_2);
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