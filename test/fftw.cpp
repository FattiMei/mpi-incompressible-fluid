#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h> // sudo apt install libfftw3-dev
// g++ -o laplacian3d tes.cpp -lfftw3 -lm


constexpr double PI = 3.141592653589793;
constexpr int N = 100; // Number of points in each dimension

using namespace std;

// Helper function to compute eigenvalues for the Laplacian in Fourier space
double compute_eigenvalue(int index, int N) {
    return 2.0 * (cos(2.0 * PI * index / N) - 1.0);
}

int main() {
    // Step 1: Define input arrays and set up FFTW plans
    int total_points = N * N * N;

    // Allocate memory for FFTW input/output arrays
    fftw_complex *b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);
    fftw_complex *btilde = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);
    fftw_complex *xtilde = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);
    fftw_complex *x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_points);

    // Create FFTW plans
    fftw_plan forward_plan = fftw_plan_dft_3d(N, N, N, b, btilde, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_3d(N, N, N, xtilde, x, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Initialize b (the RHS) with some random values
    for (int i = 0; i < total_points; i++) {
        b[i][0] = static_cast<double>(rand()) / RAND_MAX; // Real part
        b[i][1] = 0.0;                                    // Imaginary part
    }

    // Step 2: Perform forward FFT to transform b into btilde
    fftw_execute(forward_plan);

    // Step 3: Compute xtilde in Fourier space
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int idx = i * N * N + j * N + k;

                double eigx = compute_eigenvalue(i, N);
                double eigy = compute_eigenvalue(j, N);
                double eigz = compute_eigenvalue(k, N);

                double eigenvalue_sum = eigx + eigy + eigz;

                if (i == 0 && j == 0 && k == 0) {
                    xtilde[idx][0] = 0.0;
                    xtilde[idx][1] = 0.0;
                } else {
                    xtilde[idx][0] = btilde[idx][0] / eigenvalue_sum; // Real part
                    xtilde[idx][1] = btilde[idx][1] / eigenvalue_sum; // Imaginary part
                }
            }
        }
    }

    // Step 4: Perform inverse FFT to transform xtilde back to x
    fftw_execute(backward_plan);

    // Step 5: Normalize the inverse FFT output and check the result
    vector<double> x_real(total_points, 0.0);
    double normalization_factor = 1.0 / total_points;
    for (int i = 0; i < total_points; i++) {
        x_real[i] = x[i][0] * normalization_factor; // Normalize and extract real part
        assert(fabs(x[i][1]) < 1e-6);              // Ensure imaginary part is negligible
    }

    // Check if xex and x differ by a constant
    vector<double> xex(total_points, 0.0); // Replace with your exact solution
    double constant = xex[0] - x_real[0];
    for (int i = 1; i < total_points; i++) {
        assert(fabs((xex[i] - x_real[i]) - constant) < 1e-6);
    }

    cout << "Constant difference between xex and x: " << constant << endl;

    // Cleanup FFTW resources
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(b);
    fftw_free(btilde);
    fftw_free(xtilde);
    fftw_free(x);

    return 0;
}