#ifndef PRESSURE_EQUATION_H
#define PRESSURE_EQUATION_H

#include "PressureSolverStructures.h"
#include "VelocityTensor.h"

namespace mif {
    void solve_pressure_equation_neumann(StaggeredTensor &pressure, 
                                         const VelocityTensor &velocity,
                                         PressureSolverStructures &structures);
}

#endif // PRESSURE_EQUATION_H