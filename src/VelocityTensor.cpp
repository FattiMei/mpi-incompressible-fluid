#include "VelocityTensor.h"
#include "StaggeredTensorMacros.h"

namespace mif {

VelocityTensor::VelocityTensor(const Constants &constants): 
    u(constants),
    v(constants),
    w(constants),
    components({&this->u,&this->v,&this->w}),
    constants(constants) {}

void VelocityTensor::swap_data(VelocityTensor &other) {
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *old_buffer = components[component];
    StaggeredTensor *new_buffer = (other.components[component]);

    old_buffer->swap_data(*new_buffer);

    // Recompute MPI addressing after swapping, as swapping the buffer pointers
    // is not enough to truly capture the new data layout, as MPI addressing
    // will contain invalid addresses. New addresses are computed and stored.
    old_buffer->recompute_mpi_addressing();
    new_buffer->recompute_mpi_addressing();
  }
}

void VelocityTensor::set(const VectorFunction &f, bool include_border) {
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const auto *func = f.components[component];
    tensor->set(*func, include_border);
  }
}

void VelocityTensor::apply_bc(const VectorFunction &exact_velocity) {
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const std::array<size_t, 3> sizes = tensor->sizes();
    const std::function<Real(Real, Real, Real)> &func =
        (component == 0) ? exact_velocity.f_u : (component == 1 ? exact_velocity.f_v : exact_velocity.f_w);

    // First, apply Dirichlet boundary conditions on real boundaries (not processor
    // boundaries).

    // Face 1: z=z_min
    if (constants.prev_proc_z == -1 && !constants.periodic_bc[2]) {
      if (component == 2) {
        for (size_t j = 1; j < constants.Ny - 1; j++) {
          for (size_t i = 1; i < constants.Nx - 1; i++) {
            const Real w_at_boundary = tensor->evaluate_function_at_index_unstaggered(i, j, 0, func);
            const Real du_dx =
                (u.evaluate_function_at_index(i + 1, j, 0, exact_velocity.f_u) -
                u.evaluate_function_at_index(i, j, 0, exact_velocity.f_u)) *
                constants.one_over_dx;
            const Real dv_dy =
                (v.evaluate_function_at_index(i, j + 1, 0, exact_velocity.f_v) -
                v.evaluate_function_at_index(i, j, 0, exact_velocity.f_v)) *
                constants.one_over_dy;
            w(i, j, 0) = w_at_boundary + constants.dz_over_2 * (du_dx + dv_dy);
          }
        }
      } else {
        for (size_t j = 0; j < sizes[1]; j++) {
          for (size_t i = 0; i < sizes[0]; i++) {
            (*tensor)(i, j, 0) =
                tensor->evaluate_function_at_index(i, j, 0, func);
          }
        }
      }
    }

    // Face 2: z=z_max
    if (constants.next_proc_z == -1 && !constants.periodic_bc[2]) {
      if (component == 2) {
        for (size_t j = 1; j < constants.Ny - 1; j++) {
          for (size_t i = 1; i < constants.Nx - 1; i++) {
            const Real w_at_boundary = tensor->evaluate_function_at_index_unstaggered(i, j, constants.Nz - 1, func);
            const Real du_dx = (u.evaluate_function_at_index(
                                    i + 1, j, constants.Nz - 1, exact_velocity.f_u) -
                                u.evaluate_function_at_index(
                                    i, j, constants.Nz - 1, exact_velocity.f_u)) *
                              constants.one_over_dx;
            const Real dv_dy = (v.evaluate_function_at_index(
                                    i, j + 1, constants.Nz - 1, exact_velocity.f_v) -
                                v.evaluate_function_at_index(
                                    i, j, constants.Nz - 1, exact_velocity.f_v)) *
                              constants.one_over_dy;
            w(i, j, constants.Nz_staggered - 1) =
                w_at_boundary - constants.dz_over_2 * (du_dx + dv_dy);
          }
        }
      } else {
        for (size_t j = 0; j < sizes[1]; j++) { 
          for (size_t i = 0; i < sizes[0]; i++) {
            (*tensor)(i, j, constants.Nz - 1) =
                tensor->evaluate_function_at_index(i, j, constants.Nz - 1,
                                                  func);
          }
        }
      }
    }

    // Face 3: y=y_min
    if (constants.prev_proc_y == -1 && !constants.periodic_bc[1]) {
      if (component == 1) {
        for (size_t k = 1; k < constants.Nz - 1; k++) { 
          for (size_t i = 1; i < constants.Nx - 1; i++) {
            const Real v_at_boundary = tensor->evaluate_function_at_index_unstaggered(i, 0, k, func);
            const Real du_dx =
                (u.evaluate_function_at_index(i + 1, 0, k, exact_velocity.f_u) -
                u.evaluate_function_at_index(i, 0, k, exact_velocity.f_u)) *
                constants.one_over_dx;
            const Real dw_dz =
                (w.evaluate_function_at_index(i, 0, k + 1, exact_velocity.f_w) -
                w.evaluate_function_at_index(i, 0, k, exact_velocity.f_w)) *
                constants.one_over_dz;
            v(i, 0, k) = v_at_boundary + constants.dy_over_2 * (du_dx + dw_dz);
          }
        }
      } else {
        for (size_t k = 0; k < sizes[2]; k++) {
          for (size_t i = 0; i < sizes[0]; i++) {
            (*tensor)(i, 0, k) =
                tensor->evaluate_function_at_index(i, 0, k, func);
          }
        }
      }
    }

    // Face 4: y=y_max
    if (constants.next_proc_y == -1 && !constants.periodic_bc[1]) {
      if (component == 1) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          for (size_t i = 1; i < constants.Nx - 1; i++) {
            const Real v_at_boundary = tensor->evaluate_function_at_index_unstaggered(i, constants.Ny - 1, k, func);
            const Real du_dx = (u.evaluate_function_at_index(
                                    i + 1, constants.Ny - 1, k, exact_velocity.f_u) -
                                u.evaluate_function_at_index(
                                    i, constants.Ny - 1, k, exact_velocity.f_u)) *
                              constants.one_over_dx;
            const Real dw_dz = (w.evaluate_function_at_index(
                                    i, constants.Ny - 1, k + 1, exact_velocity.f_w) -
                                w.evaluate_function_at_index(
                                    i, constants.Ny - 1, k, exact_velocity.f_w)) *
                              constants.one_over_dz;
            v(i, constants.Ny_staggered - 1, k) =
                v_at_boundary - constants.dy_over_2 * (du_dx + dw_dz);
          }
        }
      } else {
        for (size_t k = 0; k < sizes[2]; k++) {
          for (size_t i = 0; i < sizes[0]; i++) {
            (*tensor)(i, constants.Ny - 1, k) =
                tensor->evaluate_function_at_index(i, constants.Ny - 1, k,
                                                  func);
          }
        }
      }
    }

    // Face 5: x=x_min
    if (!constants.periodic_bc[0]) {
      if (component == 0) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          for (size_t j = 1; j < constants.Ny - 1; j++) {
            const Real u_at_boundary = tensor->evaluate_function_at_index_unstaggered(0, j, k, func);
            const Real dv_dy =
                (v.evaluate_function_at_index(0, j + 1, k, exact_velocity.f_v) -
                v.evaluate_function_at_index(0, j, k, exact_velocity.f_v)) *
                constants.one_over_dy;
            const Real dw_dz =
                (w.evaluate_function_at_index(0, j, k + 1, exact_velocity.f_w) -
                w.evaluate_function_at_index(0, j, k, exact_velocity.f_w)) *
                constants.one_over_dz;
            u(0, j, k) = u_at_boundary + constants.dx_over_2 * (dv_dy + dw_dz);
          }
        }
      } else {
        for (size_t k = 0; k < sizes[2]; k++) {
          for (size_t j = 0; j < sizes[1]; j++) {
            (*tensor)(0, j, k) =
                tensor->evaluate_function_at_index(0, j, k, func);
          }
        }
      }
    }

    // Face 6: x=x_max
    if (!constants.periodic_bc[0]) {
      if (component == 0) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          for (size_t j = 1; j < constants.Ny - 1; j++) {
            const Real u_at_boundary = tensor->evaluate_function_at_index_unstaggered(constants.Nx_domains, j, k, func);
            const Real dv_dy = (v.evaluate_function_at_index(
                                    constants.Nx - 1, j + 1, k, exact_velocity.f_v) -
                                v.evaluate_function_at_index(
                                    constants.Nx - 1, j, k, exact_velocity.f_v)) *
                              constants.one_over_dy;
            const Real dw_dz = (w.evaluate_function_at_index(
                                    constants.Nx - 1, j, k + 1, exact_velocity.f_w) -
                                w.evaluate_function_at_index(
                                    constants.Nx - 1, j, k, exact_velocity.f_w)) *
                              constants.one_over_dz;
            u(constants.Nx_staggered - 1, j, k) =
                u_at_boundary - constants.dx_over_2 * (dv_dy + dw_dz);
          }
        }
      } else {
        for (size_t k = 0; k < sizes[2]; k++) {
          for (size_t j = 0; j < sizes[1]; j++) {
            (*tensor)(constants.Nx - 1, j, k) =
                tensor->evaluate_function_at_index(constants.Nx - 1, j, k,
                                                  func);
          }
        }
      }
    }

    // Then, handle periodic BC fully on this processor.
    tensor->apply_periodic_bc();

    // Then, send MPI data for processor boundaries, including
    // periodic BC across processors.
    tensor->send_mpi_data(component*4);
  }

  // Finally, receive MPI data.
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    tensor->receive_mpi_data(component*4);
  }
}
} // namespace mif