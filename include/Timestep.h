//
// Created by giorgio on 10/10/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
#define MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H

#include "MomentumEquation.h"
#include "Constants.h"
#include "Tensor.h"

namespace mif {

    // Perform a single step of an explicit RK3 method for the velocity tensors, setting Dirichlet boundary conditions.
    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, const Constants &constants);

} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
