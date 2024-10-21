//
// Created by giorgio on 10/10/2024.
//

#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "VelocityTensor.h"

namespace mif {

    // Perform a single step of an explicit RK3 method for the velocity tensors,
    // setting Dirichlet boundary conditions.
    void timestep(VelocityTensor &velocity,
                  VelocityTensor &velocity_buffer1,
                  VelocityTensor &velocity_buffer2,
                  Real t_n);

} // mif

#endif // TIMESTEP_H
