//
// Created by giorgio on 10/10/2024.
//

#ifndef TIMESTEP_VELOCITY_H
#define TIMESTEP_VELOCITY_H

#include "PressureTensor.h"
#include "VelocityTensor.h"

namespace mif {

// Perform a single step of an explicit RK3 method,
// setting Dirichlet boundary conditions on the velocity and homogeneous Neumann on the pressure.
// MPI messages with tags in [0, 35] will be used.
void timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
              VelocityTensor &velocity_buffer_2, const TimeVectorFunction &exact_velocity,
              Real t_n, StaggeredTensor &pressure, StaggeredTensor &pressure_buffer, 
              PressureTensor &pressure_solver_buffer);

// Do the same things, but with non-homogeneous Neumann boundary conditions
// on the pressure.
void timestep_nhn(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
                  VelocityTensor &velocity_buffer_2, const TimeVectorFunction &exact_velocity,
                  const TimeVectorFunction &exact_pressure_gradient, Real t_n,
                  StaggeredTensor &pressure, StaggeredTensor &pressure_buffer, 
                  PressureTensor &pressure_solver_buffer);

} // namespace mif

#endif // TIMESTEP_VELOCITY_H


