#include "StaggeredTensor.h"
#include <iostream>

namespace mif {
    StaggeredTensor::StaggeredTensor(const Constants &constants, const StaggeringDirection &staggering)
      : Tensor(staggering == StaggeringDirection::x ? std::array<size_t,3>({constants.Nx_staggered, constants.Ny, constants.Nz}) : 
               (staggering == StaggeringDirection::y ? std::array<size_t,3>({constants.Nx, constants.Ny_staggered, constants.Nz}) :
               (staggering == StaggeringDirection::z ? std::array<size_t,3>({constants.Nx, constants.Ny, constants.Nz_staggered}) :
               std::array<size_t,3>({constants.Nx, constants.Ny, constants.Nz})))), 
      constants(constants), staggering(staggering), 
      prev_y_request(MPI_REQUEST_NULL), next_y_request(MPI_REQUEST_NULL), 
      prev_z_request(MPI_REQUEST_NULL), next_z_request(MPI_REQUEST_NULL),
      prev_y_slice_recv({}), next_y_slice_recv({}),
      prev_y_slice_send({}), next_y_slice_send({}) {
  if (constants.Py > 1 || constants.Pz > 1) {
    const std::array<size_t, 3> &sizes = this->sizes();
    // We treat non-zero processors by creating MPI datatypes for slices of the
    // tensor. MPI addressing is computed, and the relative information about
    // neighbouring processors, min and max addresses, etc., is stored.
    MPI_Type_contiguous(sizes[0]*sizes[2], MPI_MIF_REAL, &Slice_type_constant_y);
    MPI_Type_commit(&Slice_type_constant_y);
    MPI_Type_contiguous(sizes[0]*sizes[1], MPI_MIF_REAL, &Slice_type_constant_z);
    MPI_Type_commit(&Slice_type_constant_z);

    if (constants.prev_proc_y != -1) {
      prev_y_slice_recv = Tensor<Real, 2U, size_t>({sizes[0], sizes[2]});
      prev_y_slice_send = Tensor<Real, 2U, size_t>({sizes[0], sizes[2]});
    }
     if (constants.next_proc_y != -1) {
      next_y_slice_send = Tensor<Real, 2U, size_t>({sizes[0], sizes[2]});
      next_y_slice_recv = Tensor<Real, 2U, size_t>({sizes[0], sizes[2]});
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
    MPI_Status status;
    int outcome = MPI_Wait(&prev_z_request, &status);
    assert(outcome == MPI_SUCCESS);
    outcome = MPI_Isend(min_addr_send_z, 1, Slice_type_constant_z, constants.prev_proc_z, base_tag, MPI_COMM_WORLD, &prev_z_request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  }                                                                                                                                                                            
  if (constants.next_proc_z != -1) {
    // Send data to the "right" neighbour.  
    MPI_Status status;
    int outcome = MPI_Wait(&next_z_request, &status);
    assert(outcome == MPI_SUCCESS);
    outcome = MPI_Isend(max_addr_send_z, 1, Slice_type_constant_z, constants.next_proc_z, base_tag + 1, MPI_COMM_WORLD, &next_z_request);
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
    MPI_Status status;
    int outcome = MPI_Wait(&prev_y_request, &status);
    assert(outcome == MPI_SUCCESS);
    outcome = MPI_Isend(min_addr_send_y, 1, Slice_type_constant_y, constants.prev_proc_y, base_tag + 2, MPI_COMM_WORLD, &prev_y_request);
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
    MPI_Status status;
    int outcome = MPI_Wait(&next_y_request, &status);
    assert(outcome == MPI_SUCCESS);
    outcome = MPI_Isend(max_addr_send_y, 1, Slice_type_constant_y, constants.next_proc_y, base_tag + 3, MPI_COMM_WORLD, &next_y_request);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;
  } 
}

void StaggeredTensor::receive_mpi_data(int base_tag) {
  // After checking if the neighbouring processors are valid, we receive the data
  // to them using MPI_Recv, which is a blocking receive operation.
  // This is where we practically use previously computed MPI addressing. 
  const std::array<size_t, 3> &sizes = this->sizes();                                                                                                                                                                         
  if (constants.next_proc_z != -1) {
    // Receive data from the "right" neighbour.                                                                                                                                 
    MPI_Status status;
    int return_code = MPI_Recv(max_addr_recv_z, 1, Slice_type_constant_z, constants.next_proc_z, base_tag, MPI_COMM_WORLD, &status);
    assert(return_code == 0);
    (void) return_code;
  } 
  if (constants.prev_proc_z != -1) {                                                                                                                                 
    // Receive data from the "left" neighbour.
    MPI_Status status;
    int return_code = MPI_Recv(min_addr_recv_z, 1, Slice_type_constant_z, constants.prev_proc_z, base_tag + 1, MPI_COMM_WORLD, &status);
    assert(return_code == 0);
    (void) return_code;
  } 
  if (constants.next_proc_y != -1) {
      // Receive data from the "bottom" neighbour.  
      MPI_Status status;
      int return_code = MPI_Recv(max_addr_recv_y, 1, Slice_type_constant_y, constants.next_proc_y, base_tag + 2, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;

      // Copy it into the tensor.
      for (size_t k = 1; k < sizes[2]-1; k++) {
        for (size_t i = 1; i < sizes[0]-1; i++) {
          this->operator()(i, sizes[1] - 1, k) = next_y_slice_recv(i,k);
        }
      }
  }   
  if (constants.prev_proc_y != -1) {
      // Receive data from the "top" neighbour.  
      MPI_Status status;
      int return_code = MPI_Recv(min_addr_recv_y, 1, Slice_type_constant_y, constants.prev_proc_y, base_tag + 3, MPI_COMM_WORLD, &status);
      assert(return_code == 0);
      (void) return_code;

      // Copy it into the tensor.
      for (size_t k = 1; k < sizes[2]-1; k++) {
        for (size_t i = 1; i < sizes[0]-1; i++) {
          this->operator()(i, 0, k) = prev_y_slice_recv(i,k);
        }
      }
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
  const std::array<size_t, 3> &sizes = this->sizes();
  for (size_t i = 0; i < sizes[0]*sizes[1]*sizes[2]; i++) {
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

  for (size_t k = lower_limit; k < upper_limit_k; k++) {
    for (size_t j = lower_limit; j < upper_limit_j; j++) {
      for (size_t i = lower_limit; i < upper_limit_i; i++) {
        this->operator()(i, j, k) = evaluate_function_at_index(i, j, k, f);
      }
    }
  }
}

void StaggeredTensor::apply_periodic_bc() {
  const std::array<size_t, 3> &sizes = this->sizes();

  // x faces.
  if (constants.periodic_bc[0]) {
    // Copy the second to last x slice into the first,
    // and the second x slice into the last.
    for (size_t k = 0; k < sizes[2]; k++) {
      for (size_t j = 0; j < sizes[1]; j++) {
        this->operator()(0,j,k) = this->operator()(sizes[0]-2,j,k);
        this->operator()(sizes[0]-1,j,k) = this->operator()(1,j,k);
      }
    }
  }

  if (constants.periodic_bc[1] && constants.Py == 1) {
    // Copy the second to last y slice into the first,
    // and the second y slice into the last.
    for (size_t k = 0; k < sizes[2]; k++) {
      for (size_t i = 0; i < sizes[0]; i++) {
        this->operator()(i,0,k) = this->operator()(i,sizes[1]-2,k);
        this->operator()(i,sizes[1]-1,k) = this->operator()(i,1,k);
      }
    }
  }

  if (constants.periodic_bc[2] && constants.Pz == 1) {
    // Copy the second to last z slice into the first,
    // and the second z slice into the last.
    for (size_t j = 0; j < sizes[1]; j++) {
      for (size_t i = 0; i < sizes[0]; i++) {
        this->operator()(i,j,0) = this->operator()(i,j,sizes[2]-2);
        this->operator()(i,j,sizes[2]-1) = this->operator()(i,j,1);
      }
    }
  }
}

}
