#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <fftw3.h>
#include "Real.h"

using namespace std;


// this test solves the 1D problem laplacian(p) = ... with neumann boundary conditions with a serial algorithm
int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: <program> <number of nodes>" << std::endl;
		exit(1);
	}

	const int N = std::atoi(argv[1]);

	Real *x      = (Real*) fftw_malloc(sizeof(Real) * N);
	Real *xex    = (Real*) fftw_malloc(sizeof(Real) * N);
	Real *b      = (Real*) fftw_malloc(sizeof(Real) * N);

#if USE_DOUBLE
	fftw_plan b_to_b_in_place_plan = fftw_plan_r2r_1d(N, b, b, FFTW_REDFT00,  FFTW_ESTIMATE);
	fftw_plan x_to_x_in_place_plan = fftw_plan_r2r_1d(N, x, x, FFTW_REDFT00, FFTW_ESTIMATE);
#else
	fftwf_plan b_to_b_in_place_plan = fftwf_plan_r2r_1d(N, b, b, FFTW_REDFT00,  FFTW_ESTIMATE);
	fftwf_plan x_to_x_in_place_plan = fftwf_plan_r2r_1d(N, x, x, FFTW_REDFT00, FFTW_ESTIMATE);
#endif

	// cos(x) for x in [0, 2*pi] solves the problem (it has zero gradient at the boundary)
	// differing from the periodic case, we need to consider all the points on the boundary
	const Real h = 2.0*M_PI/(N-1);
	for (int i = 0; i < N; ++i) {
		xex[i] = std::cos(h*i);
		b[i]   = -xex[i];
	}

#if USE_DOUBLE
	fftw_execute(b_to_b_in_place_plan);
#else
	fftwf_execute(b_to_b_in_place_plan);
#endif

	x[0] = 0.0;

	for (int i = 1; i < N; ++i) {
		const Real eig = 2.0 * std::cos(M_PI * i / (N-1)) - 2.0;

		x[i] = (b[i] / eig) * (h * h);
	}


#if USE_DOUBLE
	fftw_execute(x_to_x_in_place_plan);
#else
	fftwf_execute(x_to_x_in_place_plan);
#endif

	// the normalization is different compared to the periodic case
	for (int i = 0; i < N; ++i) {
		x[i] /= 2.0 * (N-1);
	}

	const Real constant = xex[0] - x[0];
	Real maxerr = 0;

	for (int i = 0; i < N; ++i) {
		const Real err = std::abs(xex[i] - x[i] - constant);

		if (err > maxerr) {
			maxerr = err;
		}
	}

	std::cout << maxerr << std::endl;

#if USE_DOUBLE
	fftw_destroy_plan(b_to_b_in_place_plan);
	fftw_destroy_plan(x_to_x_in_place_plan);
	fftw_free(x);
	fftw_free(xex);
	fftw_free(b);
#else
	fftwf_destroy_plan(b_to_b_in_place_plan);
	fftwf_destroy_plan(x_to_x_in_place_plan);
	fftwf_free(x);
	fftwf_free(xex);
	fftwf_free(b);
#endif

	return 0;
}
