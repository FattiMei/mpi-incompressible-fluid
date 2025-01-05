#ifndef PRESSURE_GRADIENT_H
#define PRESSURE_GRADIENT_H

#include "StaggeredTensor.h"

namespace mif {

// Compute dp/dx in the staggered point (i,j,k) staggered in the x direction.
inline Real pressure_gradient_u(const StaggeredTensor &pressure,
                                const size_t i, const size_t j, const size_t k) {
    return (pressure(i,j,k) - pressure(i-1,j,k)) * pressure.constants.one_over_dx;
}

// Compute dp/dy in the staggered point (i,j,k) staggered in the y direction.
inline Real pressure_gradient_v(const StaggeredTensor &pressure,
                                const size_t i, const size_t j, const size_t k) {
    return (pressure(i,j,k) - pressure(i,j-1,k)) * pressure.constants.one_over_dy;
}

// Compute dp/dz in the staggered point (i,j,k) staggered in the z direction.
inline Real pressure_gradient_w(const StaggeredTensor &pressure,
                                const size_t i, const size_t j, const size_t k) {
    return (pressure(i,j,k) - pressure(i,j,k-1)) * pressure.constants.one_over_dz;
}

} // namespace mif

#endif // PRESSURE_GRADIENT_H
