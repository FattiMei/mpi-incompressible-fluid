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




void apply_operator(const int N, const double x[], double b[]) {
	b[0] = -2.0 * x[0] + x[1] + x[N-1];

	for (int i = 1; i < N-1; ++i) {
		b[i] = -2.0 * x[i] + x[i-1] + x[i+1];
	}

	b[N-1] = -2.0 * x[N-1] + x[N-2] + x[0];
}


int main() {
	int total_points = N;

	double *x      = (double*) fftw_malloc(sizeof(double) * total_points);
	double *xex    = (double*) fftw_malloc(sizeof(double) * total_points);
	double *b      = (double*) fftw_malloc(sizeof(double) * total_points);
	double *btilde = (double*) fftw_malloc(sizeof(double) * total_points);
	double *xtilde = (double*) fftw_malloc(sizeof(double) * total_points);

	// Create FFTW plans
	fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftw_plan xtilde_to_x_plan = fftw_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);

    double h = 2*PI/(N-1);
	for (int i = 0; i < N; ++i) {
		xex[i] = std::cos(h*i);
        b[i]   = -xex[i];
	}
    b[0] /= std::sqrt(2);
    b[N-1] /= std::sqrt(2);
	// apply_operator(N, xex, b);
	fftw_execute(b_to_btilde_plan);

	xtilde[0] = 0.0;
	xtilde[0] = 0.0;
	for (int i = 1; i < N; ++i) {
		const double eig = 2*std::cos(2 * M_PI * i / N) - 2.0;

		xtilde[i] = btilde[i] / eig;
	}


//   fftw_execute(xtilde_to_x_plan);

//   // ATTENTION! Renormalization is necessary such that IFFT . FFT == Identity
//   for (int i = 0; i < N; ++i) {
//   	x[i] /= N;
//   }
    x[0]   *= sqrt(2);
    x[N-1] *= sqrt(2);
	const double constant = xex[0] - x[0];

	for (int i = 0; i < N; ++i) {
		assert(std::abs(xex[i] - x[i] - constant) < 1e-6);
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
