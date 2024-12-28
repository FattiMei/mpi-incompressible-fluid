#include "PressureTensor.h"

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

    // Execute CODE over all points common to a PressureTensor and a StaggeredTensor.
    #define ITERATE_OVER_ALL_COMMON_POINTS(CODE) {                                          \
        const std::array<size_t, 3> &sizes = other.sizes();                                 \
        const size_t min_k = (constants.prev_proc_z == -1) ? 0: 1;                          \
        const size_t max_k = (constants.next_proc_z == -1) ? sizes[2]: sizes[2]-1;          \
        const size_t min_j = (constants.prev_proc_y == -1) ? 0: 1;                          \
        const size_t max_j = (constants.next_proc_y == -1) ? sizes[1]: sizes[1]-1;          \
                                                                                            \
        int index = 0;                                                                      \
        for (size_t k = min_k; k < max_k; k++) {                                            \
            for (size_t j = min_j; j < max_j; j++) {                                        \
                for (size_t i = 0; i < sizes[0]; i++) {                                     \
                    CODE                                                                    \
                    index++;                                                                \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
    }
    
    void PressureTensor::copy_from_staggered(const StaggeredTensor &other) {
        const Constants &constants = other.constants;
        ITERATE_OVER_ALL_COMMON_POINTS(this->operator()(index) = other(i,j,k);)
    }

    void PressureTensor::copy_to_staggered(StaggeredTensor &other, int base_tag) const {
        // Copy common points.
        const Constants &constants = other.constants;
        ITERATE_OVER_ALL_COMMON_POINTS(other(i,j,k) = this->operator()(index);)

        // Send MPI data for processor borders.
        other.send_mpi_data(base_tag);

        // Receive MPI data for processor borders.
        other.receive_mpi_data(base_tag);
    }

    void PressureTensor::print_inline() {
        for (int i = 0; i < max_size; i++) {
            std::cout << this->operator()(i) << " ";
        }
        std::cout << std::endl;
    }
}