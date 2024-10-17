//
// Created by giorgio on 10/10/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
#define MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H

#include "VelocityTensor.h"

namespace mif {

    // Perform a single step of an explicit RK3 method for the velocity tensors, setting Dirichlet boundary conditions.
    // The forcing terms have t as the first argument and x,y,z as the latter 3.
    void timestep(VelocityTensor &velocity,
                  VelocityTensor &velocity_buffer1,
                  VelocityTensor &velocity_buffer2,
                  const TimeVectorFunction &exact_velocity,
                  const TimeVectorFunction &forcing_term,
                  Real t_n);

} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
