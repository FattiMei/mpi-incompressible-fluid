#ifndef MOMENTUM_EQUATION_H
#define MOMENTUM_EQUATION_H

#include "Constants.h"
#include "Tensor.h"

namespace mif {



    inline Real calculate_momentum_rhs_u(const Tensor &u, const Tensor &v, const Tensor &w, const Constants &constants, const size_t i, const size_t j, const size_t k) {
        return -u.get(i, j, k)*(u.get(i+1, j, k)-u.get(i-1, j, k))*constants.one_over_2_dx
               -(v.get(i, j, k)+v.get(i, j-1, k)+v.get(i+1, j, k)+v.get(i+1, j-1, k))*(u.get(i, j+1, k)-u.get(i, j-1, k))*constants.one_over_8_dy
               -(w.get(i, j, k)+w.get(i, j, k-1)+w.get(i+1, j, k)+w.get(i+1, j, k-1))*(u.get(i, j, k+1)-u.get(i, j, k-1))*constants.one_over_8_dz
               +(u.get(i+1, j, k)-2*u.get(i, j, k)+u.get(i-1, j, k))*constants.one_over_dx2_Re
               +(u.get(i, j+1, k)-2*u.get(i, j, k)+u.get(i, j-1, k))*constants.one_over_dy2_Re
               +(u.get(i, j, k+1)-2*u.get(i, j, k)+u.get(i, j, k-1))*constants.one_over_dz2_Re
    }


    inline Real calculate_momentum_rhs_v(const Tensor &u, const Tensor &v, const Tensor &w, const Constants &constants, const size_t i, const size_t j, const size_t k) {
        return -(u.get(i, j, k)+u.get(i-1, j, k)+u.get(i, j+1, k)+u.get(i-1, j+1, k))*(v.get(i+1, j, k)-v.get(i-1, j, k))*constants.one_over_8_dx
               -v.get(i, j, k)*(v.get(i, j+1, k)-v.get(i, j-1, k))*constants.one_over_2_dy
               -(w.get(i, j, k)+w.get(i, j, k-1)+w.get(i, j+1, k)+w.get(i, j+1, k-1))*(v.get(i, j, k+1)-v.get(i, j, k-1))*constants.one_over_8_dz
               +(v.get(i+1, j, k)-2*v.get(i, j, k)+v.get(i-1, j, k))*constants.one_over_dx2_Re
               +(v.get(i, j+1, k)-2*v.get(i, j, k)+v.get(i, j-1, k))*constants.one_over_dy2_Re
               +(v.get(i, j, k+1)-2*v.get(i, j, k)+v.get(i, j, k-1))*constants.one_over_dz2_Re
    }


    inline Real calculate_momentum_rhs_w(const Tensor &u, const Tensor &v, const Tensor &w, const Constants &constants, const size_t i, const size_t j, const size_t k) {
        return -(u.get(i, j, k)+u.get(i-1, j, k)+u.get(i, j, k+1)+u.get(i-1, j, k+1))*(w.get(i+1, j, k)-w.get(i-1, j, k))*constants.one_over_8_dx
               -(v.get(i, j, k)+v.get(i, j-1, k)+v.get(i, j, k+1)+v.get(i, j-1, k+1))*(w.get(i, j+1, k)-w.get(i, j-1, k))*constants.one_over_8_dy
               -w.get(i, j, k)*(w.get(i, j, k+1)-w.get(i, j, k-1))*constants.one_over_2_dz
               +(w.get(i+1, j, k)-2*w.get(i, j, k)+w.get(i-1, j, k))*constants.one_over_dx2_Re
               +(w.get(i, j+1, k)-2*w.get(i, j, k)+w.get(i, j-1, k))*constants.one_over_dy2_Re
               +(w.get(i, j, k+1)-2*w.get(i, j, k)+w.get(i, j, k-1))*constants.one_over_dz2_Re
    }

} // mif

#endif;