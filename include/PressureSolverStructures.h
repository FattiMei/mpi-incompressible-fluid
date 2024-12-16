#ifndef PRESSURE_SOLVER_STRUCTURES_H
#define PRESSURE_SOLVER_STRUCTURES_H

#include "Constants.h"
#include <fftw3.h>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop

namespace mif {
    class PressureSolverStructures {
    public:
        bool boundary_conditions[3];
        C2Decomp c2d;
        Real *buffer_x; // Buffer to perform fft along the x direction.
        Real *buffer_y; // Buffer to perform fft along the y direction.
        Real *buffer_z; // Buffer to perform fft along the z direction.
        fftw_plan fft_plan_x;
        fftw_plan fft_plan_y;
        fftw_plan fft_plan_z;
        PressureSolverStructures(const Constants &constants);
        ~PressureSolverStructures();
    };
}

#endif // PRESSURE_SOLVER_STRUCTURES_H