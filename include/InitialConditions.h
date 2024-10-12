#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <functional>
#include "Constants.h"
#include "Tensor.h"

namespace mif {

    // Apply initial conditions to all points of the u component of the velocty, using the function f.
    void apply_initial_conditions_u(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants);

    // Apply initial conditions to all points of the v component of the velocty, using the function f.
    void apply_initial_conditions_v(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants);

    // Apply initial conditions to all points of the w component of the velocty, using the function f.
    void apply_initial_conditions_w(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants);

} // mif

#endif // INITIAL_CONDITIONS_H