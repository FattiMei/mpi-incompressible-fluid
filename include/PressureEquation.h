#ifndef PRESSURE_EQUATION_H
#define PRESSURE_EQUATION_H

#include "PressureSolverStructures.h"
#include "VelocityTensor.h"

namespace mif {
    void solve_pressure_equation_homogeneous_neumann(StaggeredTensor &pressure, 
                                                     const VelocityTensor &velocity,
                                                     PressureSolverStructures &structures);
    
    void solve_pressure_equation_non_homogeneous_neumann(StaggeredTensor &pressure, 
                                                         const VelocityTensor &velocity,
                                                         const VectorFunction &exact_pressure_gradient,
                                                         PressureSolverStructures &structures);
}

#endif // PRESSURE_EQUATION_H