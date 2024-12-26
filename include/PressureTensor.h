#ifndef PRESSURE_TENSOR_H
#define PRESSURE_TENSOR_H

#include "Constants.h"
#include "PressureSolverStructures.h"
#include "Tensor.h"

namespace mif {

class PressureTensor: public Tensor<Real, 3U, size_t> {
public:
    const Constants &constants;
    const std::array<size_t, 3> sizes;
    const PressureSolverStructures structures;

    PressureTensor(const Constants &constants);
};

} // mif

#endif