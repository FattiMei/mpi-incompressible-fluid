#ifndef MOMENTUM_EQUATION_H
#define MOMENTUM_EQUATION_H

#include "Constants.h"
#include "Tensor.h"

namespace mif {

    // Calculate the x component of -(u*nabla)u + 1/Re nabla^2 u at index index.
    inline Real calculate_momentum_rhs_u(const Tensor &u, const Tensor &v, const Tensor &w, const Constants &constants, const size_t index) {
        return -u[index]*(u[index+1]-u[index-1])*constants.one_over_2_dx 
               -(v[index]+v[index-constants.row_size]+v[index+1]+v[index+1-constants.row_size])*(u[index+constants.row_size]-u[index-constants.row_size])*constants.one_over_8_dy
               -(w[index]+w[index-constants.matrix_size]+w[index+1]+w[index+1-constants.matrix_size])*(u[index+constants.matrix_size]-u[index-constants.matrix_size])*constants.one_over_8_dz
               +(u[index+1]-2*u[index]+u[index-1])*constants.one_over_dx2_Re
               +(u[index+constants.row_size]-2*u[index]+u[index-constants.row_size])*constants.one_over_dy2_Re
               +(u[index+constants.matrix_size]-2*u[index]+u[index-constants.matrix_size])*constants.one_over_dz2_Re;
    }

    // Calculate the y component of -(u*nabla)u + 1/Re nabla^2 u at index index.
    inline Real calculate_momentum_rhs_v(const Tensor &u, const Tensor &v, const Tensor &w, const Constants &constants, const size_t index) {
        return -(u[index]+u[index-1]+u[index+constants.row_size]+u[index+constants.row_size-1])*(v[index+1]-v[index-1])*constants.one_over_8_dx
               -v[index]*(v[index+constants.row_size]-v[index-constants.row_size])*constants.one_over_2_dy 
               -(w[index]+w[index-constants.matrix_size]+w[index+constants.row_size]+w[index+constants.row_size-constants.matrix_size])*(v[index+constants.matrix_size]-v[index-constants.matrix_size])*constants.one_over_8_dz
               +(v[index+1]-2*v[index]+v[index-1])*constants.one_over_dx2_Re
               +(v[index+constants.row_size]-2*v[index]+v[index-constants.row_size])*constants.one_over_dy2_Re
               +(v[index+constants.matrix_size]-2*v[index]+v[index-constants.matrix_size])*constants.one_over_dz2_Re;
    }

    // Calculate the z component of -(u*nabla)u + 1/Re nabla^2 u at index index.
    inline Real calculate_momentum_rhs_w(const Tensor &u, const Tensor &v, const Tensor &w, const Constants &constants, const size_t index) {
        return -(u[index]+u[index-1]+u[index+constants.matrix_size]+u[index+constants.matrix_size-1])*(w[index+1]-w[index-1])*constants.one_over_8_dx
               -(v[index]+v[index-constants.row_size]+v[index+constants.matrix_size]+v[index+constants.matrix_size-constants.row_size])*(w[index+constants.row_size]-w[index-constants.row_size])*constants.one_over_8_dy
               -w[index]*(w[index+constants.matrix_size]-w[index-constants.matrix_size])*constants.one_over_2_dz
               +(w[index+1]-2*w[index]+w[index-1])*constants.one_over_dx2_Re
               +(w[index+constants.row_size]-2*w[index]+w[index-constants.row_size])*constants.one_over_dy2_Re
               +(w[index+constants.matrix_size]-2*w[index]+w[index-constants.matrix_size])*constants.one_over_dz2_Re;
    }

} // mif

#endif;