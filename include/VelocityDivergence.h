#ifndef VELOCITY_DIVERGENCE_H
#define VELOCITY_DIVERGENCE_H

#include "VelocityTensor.h"

namespace mif {

// Calculate the divergence of the velocity at the pressure point (i,j,k), assuming it is an internal point.
inline Real calculate_velocity_divergence(const VelocityTensor &velocity,
                                          const size_t i, const size_t j, const size_t k) {
    // Note 1: tensors are staggered back by half an index, so u(i+1, j, k) - u(i, j, k) is 
    // centered in i,j,k in pressure points.
    // Note 2: since the distance between the two points is dx and not 2dx, the denominator is just dx.
    const Constants &constants = velocity.constants;
    const Real du_dx = (velocity.u(i+1, j, k) - velocity.u(i, j, k)) * constants.one_over_dx;
    const Real dv_dy = (velocity.v(i, j+1, k) - velocity.v(i, j, k)) * constants.one_over_dy;
    const Real dw_dz = (velocity.w(i, j, k+1) - velocity.w(i, j, k)) * constants.one_over_dz;
    
    return du_dx + dv_dy + dw_dz;
}

}

#endif // VELOCITY_DIVERGENCE_H