#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>


constexpr double PI = 3.141592653589793;


using namespace std;


// This test the FFT method to solve the problem Ax=b
// @MatteoLeone wants to solve a real problem Laplacian(p) = f
int main() {
	const int N = std::atoi(argv[1]);

	double *x      = (double*) fftw_malloc(sizeof(double) * N);
	double *xex    = (double*) fftw_malloc(sizeof(double) * N);
	double *b      = (double*) fftw_malloc(sizeof(double) * N);
	double *btilde = (double*) fftw_malloc(sizeof(double) * N);
	double *xtilde = (double*) fftw_malloc(sizeof(double) * N);

	fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftw_plan xtilde_to_x_plan = fftw_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);

	const double h = 2.0*PI/(N);
	for (int i = 0; i < N; ++i) {
		xex[i] = std::cos(h*i);
		b[i]   = -xex[i];
	}

	fftw_execute(b_to_btilde_plan);

	xtilde[0] = 0.0;

	for (int i = 1; i < N; ++i) {
		const double eig = 2*std::cos(2 * M_PI * i / N) - 2.0;

		xtilde[i] = (btilde[i] / eig) * (h * h);
	}


	fftw_execute(xtilde_to_x_plan);

	// ATTENTION! Renormalization is necessary such that IFFT . FFT == Identity
	for (int i = 0; i < N; ++i) {
		x[i] /= N;
	}

	const double constant = xex[0] - x[0];

	for (int i = 0; i < N; ++i) {
		assert(std::abs(xex[i] - x[i] - constant) < 1e-6);
	}
	std::cout << maxerr << std::endl;

	fftw_destroy_plan(b_to_btilde_plan);
	fftw_destroy_plan(xtilde_to_x_plan);

	fftw_free(x);
	fftw_free(xex);
	fftw_free(b);
	fftw_free(btilde);
	fftw_free(xtilde);

	return 0;
}
