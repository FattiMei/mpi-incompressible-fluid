#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>
#include <random>
#include <mpi.h>
#include "../deps/2Decomp_C/C2Decomp.hpp"

constexpr double PI = 3.141592653589793;
constexpr int N = 4;

using namespace std;

int rand_gen()
{
	std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister RNG

    // Define the range for the random numbers
    int lower = 0;
    int upper = 100;

    // Create a distribution in the range [lower, upper]
    std::uniform_int_distribution<> dist(lower, upper);

    // Generate and print a random number
    return dist(gen);

}

inline double compute_eigenvalue_periodic(int index, int N) {
	return 2.0 * (cos(2.0 * PI * index / N) - 1.0);
}


void apply_operator(const int N, const fftw_complex x[], fftw_complex b[]) {
	b[0][0] = -2.0 * x[0][0] + x[1][0] + x[N-1][0];

	for (int i = 1; i < N-1; ++i) {
		b[i][0] = -2.0 * x[i][0] + x[i-1][0] + x[i+1][0];
	}

	b[N-1][0] = -2.0 * x[N-1][0] + x[N-2][0] + x[0][0];
}

int mod(int x, int N) {
    return (x % N + N) % N; // Ensure non-negative modulo
}


int main(int argc, char *argv[]) {

    int ierr, totRank, mpiRank;

    ierr = MPI_Init( &argc, &argv);

    //Get the number of processes
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);

    //Get the local rank
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);



    if(!mpiRank){
        cout << endl;
        cout << "-------------------" << endl;
    	cout << " C2Decomp Testing " << endl;
    	cout << "-------------------" << endl;
    	cout << endl;

    }

	int total_points = N;

	int size = N * N * N;

    // Initialize a 6D array using std::vector
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> A(
        N, std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>(
            N, std::vector<std::vector<std::vector<std::vector<int>>>>(
                N, std::vector<std::vector<std::vector<int>>>(
                    N, std::vector<std::vector<int>>(
                        N, std::vector<int>(N, 0.0))))));

    // Fill the array
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                // Derivative wrt x
                A[i][j][k][i][j][k] += -2;
                A[i][j][k][mod(i - 1, N)][j][k] += 1;
                A[i][j][k][mod(i + 1, N)][j][k] += 1;

                // Derivative wrt y
                A[i][j][k][i][j][k] += -2;
                A[i][j][k][i][mod(j - 1, N)][k] += 1;
                A[i][j][k][i][mod(j + 1, N)][k] += 1;

                // Derivative wrt z
                A[i][j][k][i][j][k] += -2;
                A[i][j][k][i][j][mod(k - 1, N)] += 1;
                A[i][j][k][i][j][mod(k + 1, N)] += 1;
            }
        }
    }

    // Reshape into 2D array
    std::vector<std::vector<double>> flatA(size, std::vector<double>(size, 0.0));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int i2 = 0; i2 < N; ++i2) {
                    for (int j2 = 0; j2 < N; ++j2) {
                        for (int k2 = 0; k2 < N; ++k2) {
                            int index1 = i * N * N + j * N + k;
                            int index2 = i2 * N * N + j2 * N + k2;
                            flatA[index1][index2] = A[i][j][k][i2][j2][k2];
                        }
                    }
                }
            }
        }
    }

	std::vector<int> Uex;
	for(int i = 0; i < size; ++i){
		Uex.push_back(rand_gen());
	}

	std::vector<int> b(size, 0);

	for (size_t i = 0; i < flatA.size(); ++i) {
        for (size_t j = 0; j < flatA[i].size(); ++j) {
            b[i] += flatA[i][j] * Uex[j];
        }
    }

    std::vector<std::vector<std::vector<double>>> b_view(
        N, std::vector<std::vector<double>>(
               N, std::vector<double>(N, 0.0)));

    // Populate the 3D view
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                b_view[i][j][k] = b[i * N * N + j * N + k];
            }
        }
    }

    int pRow = 0, pCol = 0;
    bool periodicBC[3] = {true, true, true};

    if(!mpiRank) cout << "initializing " << endl;
    C2Decomp *c2d = new C2Decomp(N, N, N, pRow, pCol, periodicBC);
    if(!mpiRank) cout << "done initializing " << endl;

    fftw_complex *x      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size); // array of 2 double pointer 
    fftw_complex *xex    = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
	double *b1      = (double*) fftw_malloc(sizeof(double) * size);
	
	fftw_complex *xtilde = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);

    auto index = [N](int i, int j, int k) {
        return i * N * N + j * N + k;
    };

    
    // Print elements in 3D view
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                b1[index(i, j, k)] = b[index(i, j, k)];
            }
        }
    }

    //apply_operator(size, xex, b1);


    double *btilde1 = (double*) fftw_malloc(sizeof(double) * size);
    

    fftw_complex *temp1      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *temp2      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);


    fftw_plan b_to_btilde_plan = fftw_plan_dft_1d(N, temp1, temp2, FFTW_FORWARD,  FFTW_ESTIMATE);
    
    for (int kk = 0; kk < 3; kk++){
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j){
                for (int k = 0; k < N; ++k) {
                    temp1[k][0] = b1[index(i, j, k)];
                }
                fftw_execute(b_to_btilde_plan);
                for (int k = 0; k < N; ++k) {
                    btilde1[index(i, j, k)] = temp2[k][0];
                }
            }
        }
        if (kk == 0)
            c2d->transposeX2Y(btilde1, btilde1);
        else if (kk == 1)
            c2d->transposeY2Z(btilde1, btilde1);
    }
    
    cout << "And the final matrix is: " << endl;

    for (int i=0; i< N; i++){
        for (int j=0; j< N; j++){
            for (int k=0; k< N; k++){
                cout << btilde1[index(i, j, k)] << " ";
            }
        }
    }
    cout << endl;

    

    // Create FFTW plans
	
	fftw_plan xtilde_to_x_plan; //= fftw_plan_dft_1d(N, xtilde, x, 1, FFTW_ESTIMATE);

    


    /*
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
	fftw_free(xtilde);*/

    // c2d->deallocXYZ(u1);
    // c2d->deallocXYZ(u2);
    // c2d->deallocXYZ(u3);



    //Now lets kill MPI
    MPI_Finalize();

	return 0;
}
