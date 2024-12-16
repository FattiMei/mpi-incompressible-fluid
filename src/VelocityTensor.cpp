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

// Set the vector function without including the border.
void VelocityTensor::set_initial(const VectorFunction &f) {
  set(f, false);
}

void VelocityTensor::apply_all_dirichlet_bc(const VectorFunction &exact_velocity) {
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const std::array<size_t, 3> sizes = tensor->sizes();
    const std::function<Real(Real, Real, Real)> &func =
        (component == 0) ? exact_velocity.f_u : (component == 1 ? exact_velocity.f_v : exact_velocity.f_w);

    // Face 1: z=0
    if (constants.prev_proc_z != -1) {
      MPI_Status status;
      int return_code = MPI_Recv(tensor->min_addr_recv_z, 1, tensor->Slice_type_constant_z, constants.prev_proc_z, component*4 + 1, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;
    } else if (component == 2) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
        for (size_t j = 1; j < constants.Ny - 1; j++) {
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
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t j = 0; j < sizes[1]; j++) {
          (*tensor)(i, j, 0) =
              tensor->evaluate_function_at_index(i, j, 0, func);
        }
      }
    }

    // Face 2: z=z_max
    if (constants.next_proc_z != -1) {
      MPI_Status status;
      int return_code = MPI_Recv(tensor->max_addr_recv_z, 1, tensor->Slice_type_constant_z, constants.next_proc_z, component*4, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;
    } else if (component == 2) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
        for (size_t j = 1; j < constants.Ny - 1; j++) {
          const Real w_at_boundary = tensor->evaluate_function_at_index_unstaggered(i, j, constants.Nz_staggered - 1, func);
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
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t j = 0; j < sizes[1]; j++) {
          (*tensor)(i, j, constants.Nz - 1) =
              tensor->evaluate_function_at_index(i, j, constants.Nz - 1,
                                                 func);
        }
      }
    }

    // From now on, notice the extra condition for MPI communication. If the
    // neighbouring processor is valid, we receive data from it, instead of
    // applying the usual Dirichlet boundary conditions. This effectively
    // treats the neighbouring processor as a Dirichlet boundary.

    // Face 3: y=0
    if (constants.prev_proc_y != -1) {
      // Receive the data.
      MPI_Status status;
      int return_code = MPI_Recv(tensor->min_addr_recv_y, 1, tensor->Slice_type_constant_y, constants.prev_proc_y, component*4 + 3, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;

      // Copy it into the tensor.
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          tensor->operator()(i, 0, k) = tensor->prev_y_slice_recv(i,k);
        }
      }
    } else if (component == 1) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
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
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(i, 0, k) =
              tensor->evaluate_function_at_index(i, 0, k, func);
        }
      }
    }

    // Face 4: y=y_max
    if (constants.next_proc_y != -1) {
      // Receive the data.
      MPI_Status status;
      int return_code = MPI_Recv(tensor->max_addr_recv_y, 1, tensor->Slice_type_constant_y, constants.next_proc_y, component*4 + 2, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;

      // Copy it into the tensor.
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          tensor->operator()(i, sizes[1] - 1, k) = tensor->next_y_slice_recv(i,k);
        }
      }
    } else if (component == 1) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real v_at_boundary = tensor->evaluate_function_at_index_unstaggered(i, constants.Ny_staggered - 1, k, func);
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
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(i, constants.Ny - 1, k) =
              tensor->evaluate_function_at_index(i, constants.Ny - 1, k,
                                                 func);
        }
      }
    }

    // Face 5: x=0
    if (component == 0) {
      for (size_t j = 1; j < constants.Ny - 1; j++) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
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
      for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(0, j, k) =
              tensor->evaluate_function_at_index(0, j, k, func);
        }
      }
    }

    // Face 6: x=x_max
    if (component == 0) {
      for (size_t j = 1; j < constants.Ny - 1; j++) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real u_at_boundary = tensor->evaluate_function_at_index_unstaggered(constants.Nx_staggered - 1, j, k, func);
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
      for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(constants.Nx - 1, j, k) =
              tensor->evaluate_function_at_index(constants.Nx - 1, j, k,
                                                 func);
        }
      }
    }
  }
}
} // namespace mif