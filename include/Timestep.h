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
    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, 
                  Tensor<> &u_buffer1, Tensor<> &v_buffer1, Tensor<> &w_buffer1, 
                  Tensor<> &u_buffer2, Tensor<> &v_buffer2, Tensor<> &w_buffer2, 
                  Tensor<> &u_buffer3, Tensor<> &v_buffer3, Tensor<> &w_buffer3, 
                  const std::function<Real(Real, Real, Real)> &forcing_term_u,
                  const std::function<Real(Real, Real, Real)> &forcing_term_v,
                  const std::function<Real(Real, Real, Real)> &forcing_term_w,
                  const Constants &constants);

} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
