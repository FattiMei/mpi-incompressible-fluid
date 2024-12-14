#include "VelocityTensor.h"
#include <iostream>
#include "ManufacturedVelocity.h"
#include "VelocityTensorMacros.h"

namespace mif {

StaggeredTensor::StaggeredTensor(const std::array<size_t, 3> &in_dimensions, const Constants &constants)
      : Tensor(in_dimensions), constants(constants), prev_y_slice_recv({}), next_y_slice_recv({}),
      prev_y_slice_send({}), next_y_slice_send({}) {
  if (constants.Px > 1 || constants.Py > 1) {
    // We treat non-zero processors by creating MPI datatypes for slices of the
    // tensor. MPI addressing is computed, and the relative information about
    // neighbouring processors, min and max addresses, etc., is stored.
    MPI_Type_contiguous(in_dimensions[1]*in_dimensions[2], MPI_MIF_REAL, &Slice_type_constant_x);
    MPI_Type_commit(&Slice_type_constant_x);
    MPI_Type_contiguous(in_dimensions[0]*in_dimensions[2], MPI_MIF_REAL, &Slice_type_constant_y);
    MPI_Type_commit(&Slice_type_constant_y);

    if (constants.prev_proc_y != -1) {
      prev_y_slice_recv = Tensor<Real, 2U, size_t>({in_dimensions[0], in_dimensions[2]});
      prev_y_slice_send = Tensor<Real, 2U, size_t>({in_dimensions[0], in_dimensions[2]});
    }
     if (constants.next_proc_y != -1) {
      next_y_slice_send = Tensor<Real, 2U, size_t>({in_dimensions[0], in_dimensions[2]});
      next_y_slice_recv = Tensor<Real, 2U, size_t>({in_dimensions[0], in_dimensions[2]});
    }

    recompute_mpi_addressing();
  }
}

void StaggeredTensor::recompute_mpi_addressing() {
  // Update the addresses used to send and receive data to and from neighbouring
  // processors.
  const std::array<size_t, 3> &sizes = this->sizes();

  // Px addresses.
  min_addr_recv_x = raw_data();
  max_addr_recv_x = static_cast<Real *>(min_addr_recv_x) + sizes[1]*sizes[2]*(sizes[0]-1);
  min_addr_send_x = static_cast<Real *>(min_addr_recv_x) + sizes[1]*sizes[2];
  max_addr_send_x = static_cast<Real *>(max_addr_recv_x) - sizes[1]*sizes[2];

  // Py addresses.
  if (constants.prev_proc_y != -1) {
    min_addr_recv_y = prev_y_slice_recv.raw_data();
    min_addr_send_y = prev_y_slice_send.raw_data();
  }
  if (constants.next_proc_y != -1) {
    max_addr_recv_y = next_y_slice_recv.raw_data();
    max_addr_send_y = next_y_slice_send.raw_data();
  }
}

void StaggeredTensor::send_mpi_data(int base_tag) {
  const std::array<size_t, 3> &sizes = this->sizes();
  
  // After checking if the neighbouring processors are valid, we send the data
  // to them using MPI_Isend, which is a non-blocking send operation.
  // This is where we practically use previously computed MPI addressing.
  if (constants.prev_proc_x != -1) {                                                                                                                                 
    // Send data to the "left" neighbour.
    MPI_Request request;                                                                                                                                                 
    int outcome = MPI_Isend(min_addr_send_x, 1, Slice_type_constant_x, constants.prev_proc_x, base_tag, MPI_COMM_WORLD, &request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  }                                                                                                                                                                            
  if (constants.next_proc_x != -1) {
    // Send data to the "right" neighbour.                                                                                                                                 
    MPI_Request request;                                                                                                                                                       
    int outcome = MPI_Isend(max_addr_send_x, 1, Slice_type_constant_x, constants.next_proc_x, base_tag + 1, MPI_COMM_WORLD, &request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  } 
  if (constants.prev_proc_y != -1) {
    // Copy data into the buffer.
    for (size_t i = 0; i < sizes[0]; i++) {
      for (size_t k = 0; k < sizes[2]; k++)  {
        prev_y_slice_send(i, k) = this->operator()(i, 1, k);
      }
    }

    // Send data to the "top" neighbour.                                                                                                                                 
    MPI_Request request;                                                                                                                                                       
    int outcome = MPI_Isend(min_addr_send_y, 1, Slice_type_constant_y, constants.prev_proc_y, base_tag + 2, MPI_COMM_WORLD, &request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  }                                                                                                                                                                            
  if (constants.next_proc_y != -1) {
    // Copy data into the buffer.
    for (size_t i = 0; i < sizes[0]; i++) {
      for (size_t k = 0; k < sizes[2]; k++)  {
        next_y_slice_send(i, k) = this->operator()(i, sizes[1]-2, k);
      }
    }

    // Send data to the "bottom" neighbour.                                                                                                                                 
    MPI_Request request;                                                                                                                                                       
    int outcome = MPI_Isend(max_addr_send_y, 1, Slice_type_constant_y, constants.next_proc_y, base_tag + 3, MPI_COMM_WORLD, &request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  } 
}

void StaggeredTensor::print() const {
  const std::array<size_t, 3> &sizes = this->sizes();
  for (size_t i = 0; i < sizes[0]; i++) {
    for (size_t j = 0; j < sizes[1]; j++) {
      for (size_t k = 0; k < sizes[2]; k++) {
        std::cout << this->operator()(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void StaggeredTensor::print(const std::function<bool(Real)> &filter) const {
  const std::array<size_t, 3> &sizes = this->sizes();
  for (size_t i = 0; i < sizes[0]; i++) {
    for (size_t j = 0; j < sizes[1]; j++) {
      for (size_t k = 0; k < sizes[2]; k++) {
        const Real value = this->operator()(i, j, k);
        if (filter(value)) {
          std::cout << "(" << i << "," << j << "," << k << "): " << value
                    << std::endl;
        }
      }
    }
  }
  std::cout << std::endl;
}

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
  const size_t lower_limit = include_border ? 0 : 1;
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const std::array<size_t, 3> &sizes = tensor->sizes();
    const auto *func = f.components[component];
    const size_t upper_limit_i = include_border ? sizes[0] : sizes[0] - 1;
    const size_t upper_limit_j = include_border ? sizes[1] : sizes[1] - 1;
    const size_t upper_limit_k = include_border ? sizes[2] : sizes[2] - 1;

    for (size_t i = lower_limit; i < upper_limit_i; i++) {
      for (size_t j = lower_limit; j < upper_limit_j; j++) {
        for (size_t k = lower_limit; k < upper_limit_k; k++) {
          (*tensor)(i, j, k) =
              tensor->evaluate_function_at_index(i, j, k, *func);
        }
      }
    }
  }
}

// Set the vector function without including the border.
void VelocityTensor::set_initial(const VectorFunction &f) {
  set(f, false);
}

void VelocityTensor::apply_all_dirichlet_bc(Real time) {
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const std::array<size_t, 3> sizes = tensor->sizes();
    const std::function<Real(Real, Real, Real, Real)> &func =
        (component == 0) ? u_exact : (component == 1 ? v_exact : w_exact);

    // Face 1: z=0
    if (component == 2) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
        for (size_t j = 1; j < constants.Ny - 1; j++) {
          const Real w_at_boundary = tensor->evaluate_function_at_index_unstaggered(time, i, j, 0, func);
          const Real du_dx =
              (u.evaluate_function_at_index(time, i + 1, j, 0, u_exact) -
               u.evaluate_function_at_index(time, i, j, 0, u_exact)) *
              constants.one_over_dx;
          const Real dv_dy =
              (v.evaluate_function_at_index(time, i, j + 1, 0, v_exact) -
               v.evaluate_function_at_index(time, i, j, 0, v_exact)) *
              constants.one_over_dy;
          w(i, j, 0) = w_at_boundary + constants.dz_over_2 * (du_dx + dv_dy);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t j = 0; j < sizes[1]; j++) {
          (*tensor)(i, j, 0) =
              tensor->evaluate_function_at_index(time, i, j, 0, func);
        }
      }
    }

    // Face 2: z=z_max
    if (component == 2) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
        for (size_t j = 1; j < constants.Ny - 1; j++) {
          const Real w_at_boundary = tensor->evaluate_function_at_index_unstaggered(time, i, j, constants.Nz_staggered - 1, func);
          const Real du_dx = (u.evaluate_function_at_index(
                                  time, i + 1, j, constants.Nz - 1, u_exact) -
                              u.evaluate_function_at_index(
                                  time, i, j, constants.Nz - 1, u_exact)) *
                             constants.one_over_dx;
          const Real dv_dy = (v.evaluate_function_at_index(
                                  time, i, j + 1, constants.Nz - 1, v_exact) -
                              v.evaluate_function_at_index(
                                  time, i, j, constants.Nz - 1, v_exact)) *
                             constants.one_over_dy;
          w(i, j, constants.Nz_staggered - 1) =
              w_at_boundary - constants.dz_over_2 * (du_dx + dv_dy);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t j = 0; j < sizes[1]; j++) {
          (*tensor)(i, j, constants.Nz - 1) =
              tensor->evaluate_function_at_index(time, i, j, constants.Nz - 1,
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
          const Real v_at_boundary = tensor->evaluate_function_at_index_unstaggered(time, i, 0, k, func);
          const Real du_dx =
              (u.evaluate_function_at_index(time, i + 1, 0, k, u_exact) -
               u.evaluate_function_at_index(time, i, 0, k, u_exact)) *
              constants.one_over_dx;
          const Real dw_dz =
              (w.evaluate_function_at_index(time, i, 0, k + 1, w_exact) -
               w.evaluate_function_at_index(time, i, 0, k, w_exact)) *
              constants.one_over_dz;
          v(i, 0, k) = v_at_boundary + constants.dy_over_2 * (du_dx + dw_dz);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(i, 0, k) =
              tensor->evaluate_function_at_index(time, i, 0, k, func);
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
          const Real v_at_boundary = tensor->evaluate_function_at_index_unstaggered(time, i, constants.Ny_staggered - 1, k, func);
          const Real du_dx = (u.evaluate_function_at_index(
                                  time, i + 1, constants.Ny - 1, k, u_exact) -
                              u.evaluate_function_at_index(
                                  time, i, constants.Ny - 1, k, u_exact)) *
                             constants.one_over_dx;
          const Real dw_dz = (w.evaluate_function_at_index(
                                  time, i, constants.Ny - 1, k + 1, w_exact) -
                              w.evaluate_function_at_index(
                                  time, i, constants.Ny - 1, k, w_exact)) *
                             constants.one_over_dz;
          v(i, constants.Ny_staggered - 1, k) =
              v_at_boundary - constants.dy_over_2 * (du_dx + dw_dz);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(i, constants.Ny - 1, k) =
              tensor->evaluate_function_at_index(time, i, constants.Ny - 1, k,
                                                 func);
        }
      }
    }

    // Face 5: x=0
    if (constants.prev_proc_x != -1) {
      MPI_Status status;
      int return_code = MPI_Recv(tensor->min_addr_recv_x, 1, tensor->Slice_type_constant_x, constants.prev_proc_x, component*4 + 1, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;
    } else if (component == 0) {
      for (size_t j = 1; j < constants.Ny - 1; j++) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real u_at_boundary = tensor->evaluate_function_at_index_unstaggered(time, 0, j, k, func);
          const Real dv_dy =
              (v.evaluate_function_at_index(time, 0, j + 1, k, v_exact) -
               v.evaluate_function_at_index(time, 0, j, k, v_exact)) *
              constants.one_over_dy;
          const Real dw_dz =
              (w.evaluate_function_at_index(time, 0, j, k + 1, w_exact) -
               w.evaluate_function_at_index(time, 0, j, k, w_exact)) *
              constants.one_over_dz;
          u(0, j, k) = u_at_boundary + constants.dx_over_2 * (dv_dy + dw_dz);
        }
      }
    } else {
      for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(0, j, k) =
              tensor->evaluate_function_at_index(time, 0, j, k, func);
        }
      }
    }

    // Face 6: x=x_max
    if (constants.next_proc_x != -1) {
      MPI_Status status;
      int return_code = MPI_Recv(tensor->max_addr_recv_x, 1, tensor->Slice_type_constant_x, constants.next_proc_x, component*4, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;
    } else if (component == 0) {
      for (size_t j = 1; j < constants.Ny - 1; j++) {
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real u_at_boundary = tensor->evaluate_function_at_index_unstaggered(time, constants.Nx_staggered - 1, j, k, func);
          const Real dv_dy = (v.evaluate_function_at_index(
                                  time, constants.Nx - 1, j + 1, k, v_exact) -
                              v.evaluate_function_at_index(
                                  time, constants.Nx - 1, j, k, v_exact)) *
                             constants.one_over_dy;
          const Real dw_dz = (w.evaluate_function_at_index(
                                  time, constants.Nx - 1, j, k + 1, w_exact) -
                              w.evaluate_function_at_index(
                                  time, constants.Nx - 1, j, k, w_exact)) *
                             constants.one_over_dz;
          u(constants.Nx_staggered - 1, j, k) =
              u_at_boundary - constants.dx_over_2 * (dv_dy + dw_dz);
        }
      }
    } else {
      for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(constants.Nx - 1, j, k) =
              tensor->evaluate_function_at_index(time, constants.Nx - 1, j, k,
                                                 func);
        }
      }
    }
  }
}
} // namespace mif