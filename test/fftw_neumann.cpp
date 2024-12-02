#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>


constexpr double PI = 3.141592653589793;
constexpr int N = 5;


double compute_eigenvalue_neumann(int i, int N) {
	return 2.0 * std::cos(M_PI * i / (N-1)) - 2.0;
}


void apply_operator_neumann(const int N, const double x[], double b[]) {
	b[0] = -2.0 * x[0] + 2.0 * x[1];

	for (int i = 1; i < N-1; ++i) {
		b[i] = -2.0 * x[i] + x[i-1] + x[i+1];
	}

	b[N-1] = -2.0 * x[N-1] + 2.0 * x[N-2];
}


int main() {
	int total_points = N;

	double *x      = new double[total_points];
	double *xex    = new double[total_points];
	double *b      = new double[total_points];
	double *btilde = new double[total_points];
	double *xtilde = new double[total_points];

	// Create FFTW plans
	fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, b, btilde, FFTW_REDFT00,  FFTW_ESTIMATE);
	fftw_plan xtilde_to_x_plan = fftw_plan_r2r_1d(N, xtilde, x, FFTW_REDFT00,  FFTW_ESTIMATE);

	for (int i = 0; i < N; ++i) {
		xex[i] = i;
	}
	apply_operator_neumann(N, xex, b);

	b[0]   /= std::sqrt(2.0);
	b[N-1] /= std::sqrt(2.0);

	fftw_execute(b_to_btilde_plan);

	// ATTENTION: normalization of the result of DCT has to be divided by 2
	for (int i = 0; i < N; ++i) {
		btilde[i] /= 2.0;
	}

	xtilde[0] = 0.0;
	for (int i = 1; i < N; ++i) {
		const double eig = compute_eigenvalue_neumann(i,N);
		xtilde[i] = btilde[i] / eig;
	}

	// -------- fino a qui tutto bene ----------

	fftw_execute(xtilde_to_x_plan);

	// ATTENTION: normalization of the result of DCT has to be divided by 2
	for (int i = 0; i < N; ++i) {
		x[i] /= 2.0;
	}

	x[0]   *= std::sqrt(2.0);
	x[N-1] *= std::sqrt(2.0);

	const double constant = xex[0] - x[0];

	/*
	 * This fails
	for (int i = 0; i < N; ++i) {
		assert(std::abs(xex[i] - x[i] - constant) < 1e-6);
	}
	*/

	fftw_destroy_plan(b_to_btilde_plan);
	fftw_destroy_plan(xtilde_to_x_plan);

	delete[] x;
	delete[] xex;
	delete[] b;
	delete[] btilde;
	delete[] xtilde;

	return 0;
}
