#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <functional>
#include "Constants.h"
#include "Tensor.h"

namespace mif {

    // Apply a Dirichlet boundary condition to all velocity components on all faces, using the function f.
    void apply_all_dirichlet_bc(Tensor &u, Tensor &v, Tensor &w, const std::function<Real(Real, Real, Real)> &f, const Constants &constants);

} // mif

#endif // BOUNDARY_CONDITIONS_H