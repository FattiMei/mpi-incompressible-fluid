#include "PressureTensor.h"

namespace mif {
    PressureTensor::PressureTensor(const Constants &constants): 
        Tensor({constants.Nx, constants.Ny_global, constants.Nz_global}),
        constants(constants), 
        sizes{constants.Nx, constants.Ny_global, constants.Nz_global},
        structures(constants) {}
}