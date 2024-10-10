//
// Created by giorgio on 07/10/24.
//
#include <vector>

#ifndef MPI_INCOMPRESSIBLE_FLUID_TENSOR_H
#define MPI_INCOMPRESSIBLE_FLUID_TENSOR_H

namespace mif {

    class Tensor {
    private:
        std::vector<double> data;
    public:
        template<typename F>
            void set(F lambda);

    };

} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_TENSOR_H
