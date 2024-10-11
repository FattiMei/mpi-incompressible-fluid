//
// Created by giorgio on 10/10/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
#define MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H

#include "MomentumEquation.h"
#include "Constants.h"
#include "Tensor.h"

namespace mif {

    void timestep(Tensor &u, Tensor &v,Tensor &w, Real dt, const Constants &constants);

} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
