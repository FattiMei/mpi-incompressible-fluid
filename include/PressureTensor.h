#ifndef PRESSURE_TENSOR_H
#define PRESSURE_TENSOR_H

#include "PressureSolverStructures.h"
#include "StaggeredTensor.h"

namespace mif {

/* 
A derived Tensor class meant to be used in the pressure solver.
There is no staggering, and no overlap between processors.
Moreover, the space allocated is the maximum space needed among
the three different reorderings.
*/  
class PressureTensor: public Tensor<Real, 1U, int> {
public:
    PressureSolverStructures &structures;
    const int max_size;

    PressureTensor(PressureSolverStructures &structures);

    // Copy data from other to this tensor.
    void copy_from_staggered(const StaggeredTensor &other);

    // Copy data from this tensor to other.
    // MPI communications will use tags in [base_tag, base_tag+3].
    void copy_to_staggered(StaggeredTensor &other, int base_tag) const;

    // Print the tensor.
    void print_inline();
};

} // mif

#endif