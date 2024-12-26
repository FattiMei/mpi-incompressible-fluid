#ifndef PRESSURE_EQUATION_H
#define PRESSURE_EQUATION_H

#include "PressureTensor.h"
#include "VelocityTensor.h"

namespace mif {
    // Solve the pressure equation with homogeneous Neumann BC. The function will use
    // tags [100, 103] for MPI communication.
    void solve_pressure_equation_homogeneous_neumann(StaggeredTensor &pressure, 
                                                     PressureTensor &pressure_buffer,
                                                     const VelocityTensor &velocity,
                                                     Real dt);
    
    // Solve the pressure equation with non-homogeneous Neumann BC. The function will use
    // tags [100, 103] for MPI communication.
    void solve_pressure_equation_non_homogeneous_neumann(StaggeredTensor &pressure, 
                                                         PressureTensor &pressure_buffer,
                                                         const VelocityTensor &velocity,
                                                         const VectorFunction &exact_pressure_gradient,
                                                         Real dt);

    // Remove a constant from the pressure to obtain the exact solution.
    void adjust_pressure(StaggeredTensor &pressure,
                         const std::function<Real(Real, Real, Real)> &exact_pressure);
}

#endif // PRESSURE_EQUATION_H