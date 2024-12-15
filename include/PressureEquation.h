#ifndef PRESSURE_EQUATION_H
#define PRESSURE_EQUATION_H

#include "StaggeredTensor.h"
#include "VelocityTensor.h"

namespace mif {
    void solve_pressure_equation_neumann(StaggeredTensor &pressure, 
                                         StaggeredTensor &pressure_tilde_buffer, 
                                         const VelocityTensor &velocity, 
                                         StaggeredTensor &b_buffer, 
                                         StaggeredTensor &b_tilde_buffer);
}

#endif // PRESSURE_EQUATION_H