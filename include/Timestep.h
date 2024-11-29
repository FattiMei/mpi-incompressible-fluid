//
// Created by giorgio on 10/10/2024.
//

#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "VelocityTensor.h"

namespace mif {

// Perform a single step of an explicit RK3 method for the velocity tensors,
// setting Dirichlet boundary conditions.
// The "base_tag" argument must be 0 for the first call, and increase by 36 each time.
void timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
              VelocityTensor &rhs_buffer, Real t_n, int base_tag);

} // namespace mif

#endif // TIMESTEP_H
