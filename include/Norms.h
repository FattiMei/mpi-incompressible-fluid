#ifndef NORMS_H
#define NORMS_H

#include "VelocityTensor.h"

namespace mif {

    Real L2Norm(const VelocityTensor &velocity,
                const VelocityTensor &exact_velocity);

    Real L1Norm(const VelocityTensor &velocity,
                const VelocityTensor &exact_velocity);

    Real LInfNorm(const VelocityTensor &velocity,
                  const VelocityTensor &exact_velocity);

} // mif

#endif // NORMS_H