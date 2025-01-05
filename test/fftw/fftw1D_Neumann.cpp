#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>
#include "Real.h"

constexpr Real PI = 3.141592653589793;
constexpr int N = 200;


Real compute_eigenvalue_neumann(int i, int N) {
	return 2.0 * std::cos(M_PI * i / (N-1)) - 2.0;
}


void apply_operator_neumann(const int N, const Real x[], Real b[]) {
	b[0] = -2.0 * x[0] + 2.0 * x[1];

	for (int i = 1; i < N-1; ++i) {
		b[i] = -2.0 * x[i] + x[i-1] + x[i+1];
	}

	b[N-1] = -2.0 * x[N-1] + 2.0 * x[N-2];
}


// This test the FFT method to solve the problem Ax=b
// @MatteoLeone wants to solve a real problem Laplacian(p) = f
int main() {
	int total_points = N;

	Real *x      = new Real[total_points];
	Real *xex    = new Real[total_points];
	Real *b      = new Real[total_points];
	Real *btilde = new Real[total_points];
	Real *xtilde = new Real[total_points];

	// Create FFTW plans
#if USE_DOUBLE
	fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftw_plan xtilde_to_x_plan = fftw_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);
#else
	fftwf_plan b_to_btilde_plan = fftwf_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftwf_plan xtilde_to_x_plan = fftwf_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);
#endif

	for (int i = 0; i < N; ++i) {
		xex[i] = std::cos(i+1);
	}
	apply_operator_neumann(N, xex, b);

#if USE_DOUBLE
	fftw_execute(b_to_btilde_plan);
#else
	fftwf_execute(b_to_btilde_plan);
#endif

	xtilde[0] = 0.0;
	for (int i = 1; i < N; ++i) {
		const Real eig = compute_eigenvalue_neumann(i,N);
		xtilde[i] = btilde[i] / eig;
	}

#if USE_DOUBLE
	fftw_execute(xtilde_to_x_plan);
#else
	fftwf_execute(xtilde_to_x_plan);
#endif

	for (int i = 0; i < N; ++i) {
		x[i] /= 2.0 * (N-1);
	}

	const Real constant = xex[0] - x[0];

	for (int i = 0; i < N; ++i) {
		assert(std::abs(xex[i] - x[i] - constant) < 1e-6);
	}
	std::cout << std::endl;

#if USE_DOUBLE
	fftw_destroy_plan(b_to_btilde_plan);
	fftw_destroy_plan(xtilde_to_x_plan);
#else
	fftwf_destroy_plan(b_to_btilde_plan);
	fftwf_destroy_plan(xtilde_to_x_plan);
#endif

	delete[] x;
	delete[] xex;
	delete[] b;
	delete[] btilde;
	delete[] xtilde;

	return 0;
}
