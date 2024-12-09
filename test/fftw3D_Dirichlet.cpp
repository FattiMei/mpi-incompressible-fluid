#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>
#include <random>
#include <mpi.h>
#include "../deps/2Decomp_C/C2Decomp.hpp"

constexpr double PI = 3.141592653589793;
constexpr int N = 3;
////// INSERT VALUE OF N

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
	return (2.0 *cos(2.0 * PI * index / N) - 2.0);
}

inline int index3D(int i, int j, int k, int N) {
    return (i * N * N) + (j * N) + k;
}

void extract_array(double arr[], int size, int start, double subarray[]){
    // Copy the elements manually
    for (int i = 0; i < size; ++i) {
        subarray[i] = arr[start + i];
    }
}

void apply_operator(const int N, const double x[], double b[]) {
	for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int idx = index3D(i, j, k, N);
                
                // Apply the operator for x-derivative
                int im1 = index3D((i - 1 + N) % N, j, k, N); // wrap around for i-1
                int ip1 = index3D((i + 1) % N, j, k, N);     // wrap around for i+1
                
                // Apply the operator for y-derivative
                int jm1 = index3D(i, (j - 1 + N) % N, k, N); // wrap around for j-1
                int jp1 = index3D(i, (j + 1) % N, k, N);     // wrap around for j+1
                
                // Apply the operator for z-derivative
                int km1 = index3D(i, j, (k - 1 + N) % N, N); // wrap around for k-1
                int kp1 = index3D(i, j, (k + 1) % N, N);     // wrap around for k+1

                // Operator sum for current element
                b[idx] = -2.0 * x[idx] * 3.0 // Center point contribution (-2 for x, y, z derivatives each)
                         + x[im1] + x[ip1]   // x-direction neighbors
                         + x[jm1] + x[jp1]   // y-direction neighbors
                         + x[km1] + x[kp1];  // z-direction neighbors
            }
        }
    }
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

	double *Uex      = (double*) fftw_malloc(sizeof(double) * size);
    double *b      = (double*) fftw_malloc(sizeof(double) * size);
	for(int i = 0; i < size; ++i){
		Uex[i] = i;
	}

    apply_operator(N, (const double*)Uex, b);

    // Sopra corretto

    int pRow = 1, pCol = 1;
    bool periodicBC[3] = {true, true, true};

    if(!mpiRank) cout << "initializing " << endl;
    C2Decomp *c2d = new C2Decomp(N, N, N, pRow, pCol, periodicBC);
    if(!mpiRank) cout << "done initializing " << endl;

    double *x      = (double*) fftw_malloc(sizeof(double) * size);
	double *btilde = (double*) fftw_malloc(sizeof(double) * size);	
	double *xtilde = (double*) fftw_malloc(sizeof(double) * size);


    double *temp1 = (double*) fftw_malloc(sizeof(double) * N);
    double *temp2 = (double*) fftw_malloc(sizeof(double) * N);
    fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, temp1, temp2, FFTW_R2HC,  FFTW_ESTIMATE);

    //apply_operator(size, xex, b1);

    // cout << "b initi: ";
    // for(int i=0; i<size; ++i)
    // {
    //     cout<< b[i]<< " ";
    // }
    // cout<<endl;
    // c2d->transposeX2Y_MajorIndex(b, b);
    // cout << "b final: ";
    // for(int i=0; i<size; ++i)
    // {
    //     cout<< b[i]<< " ";
    // }
    // cout<<endl;
    
    for (int kk = 0; kk < 3; kk++){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++){
                extract_array(b, N, i*N*N + j*N, temp1);
                fftw_execute(b_to_btilde_plan);
                for (int k = 0; k < N; k++){
                    btilde[index3D(i, j, k, N)] = temp2[k];
                }
            }
        }
        c2d->transposeX2Y_MajorIndex(btilde, btilde);
            
        b = btilde;
    }


    xtilde = btilde;

    for (int i = 0; i < N; i++) {
        double t1= compute_eigenvalue_periodic(i, N);
        for (int j = 0; j < N; j++){
            double t2= compute_eigenvalue_periodic(j, N);
            for (int k = 0; k < N; k++){
                xtilde[index3D(i, j, k, N)] /= t1 + t2 + compute_eigenvalue_periodic(k, N);
            }
        }
    }
    xtilde[0] = 0;

    // fino a qui controllato, valore xtilde corretto

    // inverse transform

    fftw_plan xtilde_to_x_plan = fftw_plan_r2r_1d(N, temp1, temp2, FFTW_HC2R,  FFTW_ESTIMATE);

    for (int kk = 0; kk < 3; kk++){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++){
                extract_array(xtilde, N, i*N*N + j*N, temp1);
               
                fftw_execute(xtilde_to_x_plan);
                for (int k = 0; k < N; k++){
                    x[index3D(i, j, k, N)] = (temp2[k]/(2.0 * (N-1)));
                }
            }
        }
        c2d->transposeX2Y_MajorIndex(x, x);
        xtilde = x;
    }

    double tyafsyxgvsib = x[0] - Uex[0];
    cout << "( ";
    for(int i=0; i<size; ++i)
    {
        assert(x[i] - Uex[i] -tyafsyxgvsib  <= 1e-15 );
        cout<< x[i] - Uex[i] - tyafsyxgvsib << "\n ";
    }
    cout << ") ";
    cout<< endl;




    
    // cout << "And the final matrix is: " << endl;
    
    // for (int i=0; i< N; i++){
    //     for (int j=0; j< N; j++){
    //         for (int k=0; k< N; k++){
    //             cout << btilde1[index(i, j, k)] << " ";
    //         }
    //     }
    // }
    // cout << endl;

    

    // Create FFTW plans


    /*
	double *x      = (double*) fftw_malloc(sizeof(double) * total_points);
	double *xex    = (double*) fftw_malloc(sizeof(double) * total_points);
	double *b      = (double*) fftw_malloc(sizeof(double) * total_points);
	double *btilde = (double*) fftw_malloc(sizeof(double) * total_points);
	double *xtilde = (double*) fftw_malloc(sizeof(double) * total_points);

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

    fftw_destroy_plan(b_to_btilde_plan);
    fftw_destroy_plan(xtilde_to_x_plan);
    fftw_cleanup();
    MPI_Finalize();

	return 0;
}
