//
// Created by giorgio on 10/10/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
#define MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
#include "MomentumEquation.h"
#include "Constants.h"
#include "Tensor.h"
void timestep(Tensor &u, Tensor &v,Tensor &w, Real dt, const Constants &constants);
#endif //MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
