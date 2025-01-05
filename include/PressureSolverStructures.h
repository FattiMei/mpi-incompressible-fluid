#ifndef PRESSURE_SOLVER_STRUCTURES_H
#define PRESSURE_SOLVER_STRUCTURES_H

#include <fftw3.h>
#include "Constants.h"
#include "Tensor.h"
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop


namespace mif {
    class PressureSolverStructures {
    public:
        bool periodic_bc[3];
        const int Nx_points;
        const int Ny_points;
        const int Nz_points;
        C2Decomp c2d;
        Real *buffer_x; // Buffer to perform fft along the x direction.
        Real *buffer_y; // Buffer to perform fft along the y direction.
        Real *buffer_z; // Buffer to perform fft along the z direction.
#if USE_DOUBLE
        const fftw_plan fft_plan_x;
        const fftw_plan fft_plan_y;
        const fftw_plan fft_plan_z;
        const fftw_plan ifft_plan_x;
        const fftw_plan ifft_plan_y;
        const fftw_plan ifft_plan_z;
#else
        const fftwf_plan fft_plan_x;
        const fftwf_plan fft_plan_y;
        const fftwf_plan fft_plan_z;
        const fftwf_plan ifft_plan_x;
        const fftwf_plan ifft_plan_y;
        const fftwf_plan ifft_plan_z;
#endif

        // The following tensor contains precomputed eigenvalues,
        // with position (j,i,k) containing 1 over the sum of the
        // respective eigenvalues, in Z storage.
        Tensor<Real, 3, int> eigenvalues; 

        PressureSolverStructures(const Constants &constants);
        ~PressureSolverStructures();
    };
}

#endif // PRESSURE_SOLVER_STRUCTURES_H