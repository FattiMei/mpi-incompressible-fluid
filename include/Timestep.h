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
    // The forcing terms have t as the first argument and x,y,z as the latter 3.
    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, 
                  Tensor<> &u_buffer1, Tensor<> &v_buffer1, Tensor<> &w_buffer1, 
                  Tensor<> &u_buffer2, Tensor<> &v_buffer2, Tensor<> &w_buffer2,
                  const std::function<Real(Real, Real, Real, Real)> &u_exact,
                  const std::function<Real(Real, Real, Real, Real)> &v_exact,
                  const std::function<Real(Real, Real, Real, Real)> &w_exact,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_u,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_v,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_w,
                  Real current_time,
                  const Constants &constants);

} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_TIMESTEP_H
