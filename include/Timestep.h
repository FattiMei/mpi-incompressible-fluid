//
// Created by giorgio on 10/10/2024.
//

#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "VelocityTensor.h"

namespace mif {

// Perform a single step of an explicit RK3 method for the velocity tensors,
// setting Dirichlet boundary conditions.
Real timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
              VelocityTensor &rhs_buffer, Real t_n, Real target_cfl,Real last_dt);

} // namespace mif

#endif // TIMESTEP_H
