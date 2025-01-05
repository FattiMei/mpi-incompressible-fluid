#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop
#include "Real.h"

constexpr Real PI = 3.141592653589793;

using namespace std;

inline Real compute_eigenvalue_periodic(int index, int N) {
	return (2.0 *cos(2.0 * PI * index /N) - 2.0);
}

inline int index3D(int i, int j, int k, int N) {
	return (i * N * N) + (j * N) + k;
}

void extract_array(Real arr[], int size, int start, Real subarray[]){
	// Copy the elements manually
	for (int i = 0; i < size; ++i) {
		subarray[i] = arr[start + i];
	}
}

int main(int argc, char *argv[]) {
	int N = std::atoi(argv[1]);
	int Nx, Ny, Nz;
	Nx = N;
	Ny = N;
	Nz = N;

	int ierr, totRank, mpiRank;

	ierr = MPI_Init( &argc, &argv);

	//Get the number of processes
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);

	//Get the local rank
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

	int total_points = N;

	int size = Nx * Ny * Nz;

	Real *Uex    = (Real*) fftw_malloc(sizeof(Real) * size);
	Real *b      = (Real*) fftw_malloc(sizeof(Real) * size);

	// Create forcing term manufactured from manufactured sulotion (u = cx*cy*cz, x, y, z = [0, 2pi])
	// REMARK: the solution vector doesn't store the last point (it has to be equal to the first one given periodic conditions)
	// this is connected to the solving algorithm that does this assumption
	Real h = 2*PI/N;
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			for (int k = 0; k < Nz; k++) {
				Uex[index3D(i,  j,  k,  N)] = std::cos(h*i) * std::cos(h*j) * std::cos(h*k);
				b[index3D(i, j, k, N)] = -3.0*Uex[index3D(i,  j,  k,  N)];
			}    
		}
	}

	// pRow*pCol = rankMPI to work properly!!
	int pRow = 1, pCol = 1;
	bool periodicBC[3] = {true, true, true};

	// if(!mpiRank) cout << "initializing " << endl;
	C2Decomp *c2d = new C2Decomp(N, N, N, pRow, pCol, periodicBC);
	// if(!mpiRank) cout << "done initializing " << endl;

	Real *x      = (Real*) fftw_malloc(sizeof(Real) * size);
	Real *btilde = (Real*) fftw_malloc(sizeof(Real) * size);	
	Real *xtilde = (Real*) fftw_malloc(sizeof(Real) * size);


	Real *temp1 = (Real*) fftw_malloc(sizeof(Real) * N);
	Real *temp2 = (Real*) fftw_malloc(sizeof(Real) * N);

#if USE_DOUBLE
	fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftw_plan xtilde_to_x_plan = fftw_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);
#else
	fftwf_plan b_to_btilde_plan = fftwf_plan_r2r_1d(N, b, btilde, FFTW_R2HC,  FFTW_ESTIMATE);
	fftwf_plan xtilde_to_x_plan = fftwf_plan_r2r_1d(N, xtilde, x, FFTW_HC2R, FFTW_ESTIMATE);
#endif

	for (int kk = 0; kk < 3; kk++){
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++){
				extract_array(b, N, i*N*N + j*N, temp1);
#if USE_DOUBLE
				fftw_execute(b_to_btilde_plan);
#else
				fftwf_execute(b_to_btilde_plan);
#endif
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
		Real t1= compute_eigenvalue_periodic(i, N);
		for (int j = 0; j < N; j++){
			Real t2= compute_eigenvalue_periodic(j, N);
			for (int k = 0; k < N; k++){
				xtilde[index3D(i, j, k, N)] /= (t1 + t2 + compute_eigenvalue_periodic(k, N) )/pow(h, 2);
			}
		}
	}
	xtilde[0] = 0;


	for (int kk = 0; kk < 3; kk++){
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++){
				extract_array(xtilde, N, i*N*N + j*N, temp1);

#if USE_DOUBLE
				fftw_execute(xtilde_to_x_plan);
#else
				fftwf_execute(xtilde_to_x_plan);
#endif
				for (int k = 0; k < N; k++){
					//x[index3D(i, j, k, N)] = (temp2[k])/(2.0 * (N-1));
					x[index3D(i, j, k, N)] = (temp2[k])/N;
				}
			}
		}
		c2d->transposeX2Y_MajorIndex(x, x);
		xtilde = x;
	}

	Real difference = x[0] - Uex[0];
	Real mazx = -2.0;
	for(int i=0; i<size; ++i)
	{
		// assert(x[i] - Uex[i] -difference  <= 1e-15 );
		Real temp = std::abs(x[i] - Uex[i] - difference );
		if (mazx < temp) mazx = temp;
	}
	cout<< mazx << endl;

#if USE_DOUBLE
	fftw_destroy_plan(b_to_btilde_plan);
	fftw_destroy_plan(xtilde_to_x_plan);
	fftw_cleanup();
#else
	fftwf_destroy_plan(b_to_btilde_plan);
	fftwf_destroy_plan(xtilde_to_x_plan);
	fftwf_cleanup();
#endif

	MPI_Finalize();

	return 0;
}
