#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>
#include "Real.h"

constexpr Real PI = 3.141592653589793;


using namespace std;


// This test the FFT method to solve the problem Ax=b
// @MatteoLeone wants to solve a real problem Laplacian(p) = f
int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: <program> <number of nodes>" << std::endl;
		exit(1);
	}

	const int N = std::atoi(argv[1]);

	Real *x      = (Real*) fftw_malloc(sizeof(Real) * N);
	Real *xex    = (Real*) fftw_malloc(sizeof(Real) * N);
	Real *b      = (Real*) fftw_malloc(sizeof(Real) * N);
	Real *btilde = (Real*) fftw_malloc(sizeof(Real) * N);
	Real *xtilde = (Real*) fftw_malloc(sizeof(Real) * N);

#if USE_DOUBLE
	fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftw_plan xtilde_to_x_plan = fftw_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);
#else
	fftwf_plan b_to_btilde_plan = fftwf_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftwf_plan xtilde_to_x_plan = fftwf_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);
#endif

	const Real h = 2.0*PI/(N);
	for (int i = 0; i < N; ++i) {
		xex[i] = std::cos(h*i);
		b[i]   = -xex[i];
	}

#if USE_DOUBLE
	fftw_execute(b_to_btilde_plan);
#else
	fftwf_execute(b_to_btilde_plan);
#endif

	xtilde[0] = 0.0;

	for (int i = 1; i < N; ++i) {
		const Real eig = 2*std::cos(2 * M_PI * i / N) - 2.0;

		xtilde[i] = (btilde[i] / eig) * (h * h);
	}


#if USE_DOUBLE
	fftw_execute(xtilde_to_x_plan);
#else
	fftwf_execute(xtilde_to_x_plan);
#endif

	// ATTENTION! Renormalization is necessary such that IFFT . FFT == Identity
	for (int i = 0; i < N; ++i) {
		x[i] /= N;
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
	fftw_destroy_plan(b_to_btilde_plan);
	fftw_destroy_plan(xtilde_to_x_plan);
	fftw_free(x);
	fftw_free(xex);
	fftw_free(b);
	fftw_free(btilde);
	fftw_free(xtilde);
#else
	fftwf_destroy_plan(b_to_btilde_plan);
	fftwf_destroy_plan(xtilde_to_x_plan);
	fftwf_free(x);
	fftwf_free(xex);
	fftwf_free(b);
	fftwf_free(btilde);
	fftwf_free(xtilde);
#endif

	return 0;
}
