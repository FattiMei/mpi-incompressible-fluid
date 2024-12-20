//
// Created by giorgio on 10/10/2024.
//

#ifndef TIMESTEP_VELOCITY_H
#define TIMESTEP_VELOCITY_H

#include "PressureSolverStructures.h"
#include "VelocityTensor.h"

namespace mif {

// Perform a single step of an explicit RK3 method,
// setting Dirichlet boundary conditions on the velocity and Neumann on the pressure.
// MPI messages with tags in [0, 35] will be used.
void timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
              VelocityTensor &rhs_buffer, const TimeVectorFunction &exact_velocity,
              const TimeVectorFunction &exact_pressure_gradient, Real t_n,
              StaggeredTensor &pressure, StaggeredTensor &pressure_buffer, 
              PressureSolverStructures &structures);

} // namespace mif

#endif // TIMESTEP_VELOCITY_H


