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
                  std::vector<std::array<Real, 3>> rhs_buffer,
                  Real t_n);

} // namespace mif

#endif // TIMESTEP_H
