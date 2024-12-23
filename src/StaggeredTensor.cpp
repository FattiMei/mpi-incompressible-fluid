#include "StaggeredTensor.h"
#include <iostream>

namespace mif {
    StaggeredTensor::StaggeredTensor(const std::array<size_t, 3> &in_dimensions, const Constants &constants)
      : Tensor(in_dimensions), constants(constants), prev_y_slice_recv({}), next_y_slice_recv({}),
      prev_y_slice_send({}), next_y_slice_send({}) {
  if (constants.Py > 1 || constants.Pz > 1) {
    // We treat non-zero processors by creating MPI datatypes for slices of the
    // tensor. MPI addressing is computed, and the relative information about
    // neighbouring processors, min and max addresses, etc., is stored.
    MPI_Type_contiguous(in_dimensions[0]*in_dimensions[2], MPI_MIF_REAL, &Slice_type_constant_y);
    MPI_Type_commit(&Slice_type_constant_y);
    MPI_Type_contiguous(in_dimensions[0]*in_dimensions[1], MPI_MIF_REAL, &Slice_type_constant_z);
    MPI_Type_commit(&Slice_type_constant_z);

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

  // Pz addresses.
  min_addr_recv_z = raw_data();
  max_addr_recv_z = static_cast<Real *>(min_addr_recv_z) + sizes[0]*sizes[1]*(sizes[2]-1);
  min_addr_send_z = static_cast<Real *>(min_addr_recv_z) + sizes[0]*sizes[1];
  max_addr_send_z = static_cast<Real *>(max_addr_recv_z) - sizes[0]*sizes[1];

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
  if (constants.prev_proc_z != -1) {                                                                                                                                 
    // Send data to the "left" neighbour.
    MPI_Request request;                                                                                                                                                 
    int outcome = MPI_Isend(min_addr_send_z, 1, Slice_type_constant_z, constants.prev_proc_z, base_tag, MPI_COMM_WORLD, &request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  }                                                                                                                                                                            
  if (constants.next_proc_z != -1) {
    // Send data to the "right" neighbour.                                                                                                                                 
    MPI_Request request;                                                                                                                                                       
    int outcome = MPI_Isend(max_addr_send_z, 1, Slice_type_constant_z, constants.next_proc_z, base_tag + 1, MPI_COMM_WORLD, &request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  } 
  if (constants.prev_proc_y != -1) {
    // Copy data into the buffer.
    for (size_t k = 0; k < sizes[2]; k++) {
      for (size_t i = 0; i < sizes[0]; i++) {
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
    for (size_t k = 0; k < sizes[2]; k++) {
      for (size_t i = 0; i < sizes[0]; i++) {
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
  for (size_t k = 0; k < sizes[2]; k++) {
    for (size_t j = 0; j < sizes[1]; j++) {
      for (size_t i = 0; i < sizes[0]; i++) {
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
  for (size_t k = 0; k < sizes[2]; k++) {
    for (size_t j = 0; j < sizes[1]; j++) {
      for (size_t i = 0; i < sizes[0]; i++) {
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

void StaggeredTensor::print_inline() const {
  for (size_t i = 0; i < constants.Nx * constants.Ny * constants.Nz; i++) {
    std::cout << this->operator()(i) << " ";
  }
  std::cout << std::endl;
}

void StaggeredTensor::set(const std::function<Real(Real, Real, Real)> &f, bool include_border) {
  const std::array<size_t, 3> &sizes = this->sizes();
  const size_t lower_limit = include_border ? 0 : 1;
  const size_t upper_limit_i = include_border ? sizes[0] : sizes[0] - 1;
  const size_t upper_limit_j = include_border ? sizes[1] : sizes[1] - 1;
  const size_t upper_limit_k = include_border ? sizes[2] : sizes[2] - 1;

  for (size_t i = lower_limit; i < upper_limit_i; i++) {
    for (size_t j = lower_limit; j < upper_limit_j; j++) {
      for (size_t k = lower_limit; k < upper_limit_k; k++) {
        this->operator()(i, j, k) = evaluate_function_at_index(i, j, k, f);
      }
    }
  }
}

}