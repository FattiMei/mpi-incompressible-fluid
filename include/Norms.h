#ifndef NORMS_H
#define NORMS_H

#include <cmath>
#include "Constants.h"
#include "Tensor.h"

namespace mif {

    Real L2Norm(const Tensor<> &U, const Tensor<> &V, 
                const Tensor<> &W, const Tensor<> &Uex, 
                const Tensor<> &Vex, const Tensor<> &Wex, 
                const Constants &c);

    Real L1Norm(const Tensor<> &U, const Tensor<> &V, 
                const Tensor<> &W, const Tensor<> &Uex, 
                const Tensor<> &Vex, const Tensor<> &Wex, 
                const mif::Constants &c);

    Real LInfNorm(const Tensor<> &U, const Tensor<> &V, 
                  const Tensor<> &W, const Tensor<> &Uex, 
                  const Tensor<> &Vex, const Tensor<> &Wex, 
                  const Constants &c);

} // mif

#endif // NORMS_H