#include "PressureTensor.h"
#include "StaggeredTensorMacros.h"

namespace mif {
    int get_max_size(const PressureSolverStructures &structures) {
        return std::max({structures.c2d.xSize[0]*structures.c2d.xSize[1]*structures.c2d.xSize[2],
                         structures.c2d.ySize[0]*structures.c2d.ySize[1]*structures.c2d.ySize[2],
                         structures.c2d.zSize[0]*structures.c2d.zSize[1]*structures.c2d.zSize[2]});
    }

    PressureTensor::PressureTensor(PressureSolverStructures &structures): 
        Tensor({get_max_size(structures)}),
        structures(structures),
        max_size(get_max_size(structures)) {}
    
    void PressureTensor::copy_from_staggered(const StaggeredTensor &other) {
        int index = 0;
        STAGGERED_TENSOR_ITERATE_OVER_ALL_OWNER_POINTS(other, this->operator()(index) = other(i,j,k); index++;)
    }

    void PressureTensor::copy_to_staggered(StaggeredTensor &other, int base_tag) const {
        // Copy common points.
        int index = 0;
        STAGGERED_TENSOR_ITERATE_OVER_ALL_OWNER_POINTS(other, other(i,j,k) = this->operator()(index); index++;)

        // Apply periodic BC that don't require MPI communication.
        other.apply_periodic_bc();

        // Send MPI data for processor borders.
        other.send_mpi_data(base_tag);

        // Receive MPI data for processor borders.
        other.receive_mpi_data(base_tag);
    }

    void PressureTensor::print_inline() const {
        for (int i = 0; i < max_size; i++) {
            std::cout << this->operator()(i) << " ";
        }
        std::cout << std::endl;
    }
}