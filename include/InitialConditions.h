#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <functional>
#include "Constants.h"
#include "Tensor.h"

namespace mif {

    // Apply initial conditions to all points of a tensor, using the function f.
    void apply_initial_conditions(Tensor &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants);

} // mif

#endif // INITIAL_CONDITIONS_H