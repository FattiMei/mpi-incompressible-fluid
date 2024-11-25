#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>

constexpr double PI = 3.141592653589793;
constexpr int N = 5;

using namespace std;

double compute_eigenvalue(int index, int N) {
	return 2.0 * (cos(2.0 * PI * index / N) - 1.0);
}


void apply_operator(const int N, const fftw_complex x[], fftw_complex b[]) {
	b[0][0] = -2.0 * x[0][0] + x[1][0] + x[N-1][0];

	for (int i = 1; i < N-1; ++i) {
		b[i][0] = -2.0 * x[i][0] + x[i-1][0] + x[i+1][0];
	}

	b[N-1][0] = -2.0 * x[N-1][0] + x[N-2][0] + x[0][0];
}


int main() {
	int total_points = N;

	fftw_complex *x      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);
	fftw_complex *xex    = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);
	fftw_complex *b      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);
	fftw_complex *btilde = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);
	fftw_complex *xtilde = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);

	// Create FFTW plans
	fftw_plan b_to_btilde_plan = fftw_plan_dft_1d(N, b, btilde, FFTW_FORWARD,  FFTW_ESTIMATE);
	fftw_plan xtilde_to_x_plan = fftw_plan_dft_1d(N, xtilde, x, 1, FFTW_ESTIMATE);

	for (int i = 0; i < N; ++i) {
		xex[i][0] = i;
	}

	apply_operator(N, xex, b);
	fftw_execute(b_to_btilde_plan);

	xtilde[0][0] = 0.0;
	xtilde[0][0] = 0.0;
	for (int i = 1; i < N; ++i) {
		const double eig = 2*std::cos(2 * M_PI * i / N) - 2.0;

		xtilde[i][0] = btilde[i][0] / eig;
		xtilde[i][1] = btilde[i][1] / eig;
	}


	fftw_execute(xtilde_to_x_plan);

	// ATTENTION! Renormalization is necessary such that IFFT . FFT == Identity
	for (int i = 0; i < N; ++i) {
		x[i][0] /= N;
		x[i][1] /= N;
	}

	const double constant = xex[0][0] - x[0][0];

	for (int i = 0; i < N; ++i) {
		assert(std::abs(xex[i][0] - x[i][0] - constant) < 1e-6);
	}

	fftw_destroy_plan(b_to_btilde_plan);
	fftw_destroy_plan(xtilde_to_x_plan);

	fftw_free(x);
	fftw_free(xex);
	fftw_free(b);
	fftw_free(btilde);
	fftw_free(xtilde);

	return 0;
}
