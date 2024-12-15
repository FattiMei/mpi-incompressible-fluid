//
// Created by giorgio on 10/10/2024.
//

#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "VelocityTensor.h"

namespace mif {

// Perform a single step of an explicit RK3 method for the velocity tensors,
// setting Dirichlet boundary conditions.
// MPI messages with tags in [0, 35] will be used.
void timestep_velocity(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
                       VelocityTensor &rhs_buffer, const TimeVectorFunction &exact_velocity, Real t_n);

} // namespace mif

#endif // TIMESTEP_H
