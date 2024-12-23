#ifndef PRESSURE_EQUATION_H
#define PRESSURE_EQUATION_H

#include "PressureSolverStructures.h"
#include "VelocityTensor.h"

namespace mif {
    void solve_pressure_equation_homogeneous_neumann(StaggeredTensor &pressure, 
                                                     const VelocityTensor &velocity,
                                                     PressureSolverStructures &structures,
                                                     Real dt);
    
    void solve_pressure_equation_non_homogeneous_neumann(StaggeredTensor &pressure, 
                                                         const VelocityTensor &velocity,
                                                         const VectorFunction &exact_pressure_gradient,
                                                         PressureSolverStructures &structures,
                                                         Real dt);

    // Remove a constant from the pressure to obtain the exact solution.
    void adjust_pressure(StaggeredTensor &pressure,
                         const std::function<Real(Real, Real, Real)> &exact_pressure);
}

#endif // PRESSURE_EQUATION_H