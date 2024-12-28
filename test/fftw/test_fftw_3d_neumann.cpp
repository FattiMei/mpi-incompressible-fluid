//TODO 
// 
// Initialize the rhs in order to match the domain decomp (Xstart ?)
// Define size local for each pencil
// Understand how to implement in order to avoid the x direction to be the slowest (Won't need exxtract array?)
// use c2d method to allocX x, b, xtilde, ...

#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>
#include <stdlib.h>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop

constexpr double PI = 3.141592653589793;

using namespace std;

inline double compute_eigenvalue_neumann(int index, int N) {
    return (2.0 *cos( PI * index / (N-1)) - 2.0);
}


void extract_array(double arr[], int size, int start, double subarray[]){
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
    (void) ierr;

    ierr = MPI_Init( &argc, &argv);

    //Get the number of processes
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);

    //Get the local rank
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);


    // pRow*pCol = totRank to work properly!!
    //int pRow = totRank, pCol = 1;
    int pRow =0 , pCol = 0;
    // True for periodic
    bool neumannBC[3] = {false, false, false};


    //Initialize 2decomp
    C2Decomp *c2d = new C2Decomp(Nx, Ny, Nz, pRow, pCol, neumannBC);

    int NxLoc = c2d->xSize[0];
    int NyLoc = c2d->xSize[1];
    int NzLoc = c2d->xSize[2];
    auto index3D =[&](int i, int j, int k) {
        return (k * NxLoc * NyLoc) + (j * NxLoc) + i;
    };
    int size = Nx * Ny * Nz;

    double *Uex   ;
    double *b     ;
    // Vectors with local informations
    c2d->allocX( Uex );
    c2d->allocX( b );



    // Create forcing term manufactured from manufactured sulotion (u =-3cx*cy*cz, x, y, z = [0, 2pi])
    double hx = 2*PI/(Nx-1) ;
    double hy = 2*PI/(Ny-1) ;
    double hz = 2*PI/(Nz-1) ;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < NyLoc; j++) {
            for (int k = 0; k < NzLoc; k++) {
                int jGlobal = j + c2d->xStart[1], kGlobal = k+c2d->xStart[2];
                Uex[index3D(i, j, k)] = std::cos(hx*i) * std::cos(hy*jGlobal) * std::cos(hz*kGlobal) ;
                b[index3D(i, j, k)] = -3.0*Uex[index3D(i, j, k)];
            }    
        }
    }
    //Check riempimento
    if(mpiRank==2){
        cout << "y0" << c2d->xStart[1] << "\n"; 
        cout << "z0" << c2d->xStart[2] << "\n"; 
        for (int j = 0; j < NyLoc; j++) {
            for (int k = 0; k < NzLoc; k++) {
                for (int i = 0; i < Nx; i++) {
                    int jGlobal = j + c2d->xStart[1], kGlobal = k+c2d->xStart[2];
                    std::cout << Uex[index3D(i, j, k)] << " \t";
                }    
                cout << "\n";
            }
            cout << " New Plane \n";
        }
    }

    double *x      = (double*) fftw_malloc(sizeof(double) * size);
    double *btilde = (double*) fftw_malloc(sizeof(double) * size);	
    double *xtilde = (double*) fftw_malloc(sizeof(double) * size);


    double *temp1 = (double*) fftw_malloc(sizeof(double) * N);
    double *temp2 = (double*) fftw_malloc(sizeof(double) * N);
    // Exucute dct type 1 along all 3 directions
    fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int kk = 0; kk < 3; kk++){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++){
                extract_array(b, N, i*N*N + j*N, temp1);
                fftw_execute(b_to_btilde_plan);
                for (int k = 0; k < N; k++){
                    btilde[index3D(i, j, k)] = temp2[k];
                }
            }
        }
        // TODO Funziona in parallelo che itero la rotazione xtoy??
        c2d->transposeX2Y_MajorIndex(btilde, btilde);

        b = btilde;
    }

    xtilde = btilde;

    for (int i = 0; i < N; i++) {
        double t1= compute_eigenvalue_neumann(i, N);
        for (int j = 0; j < N; j++){
            double t2= compute_eigenvalue_neumann(j, N);
            for (int k = 0; k < N; k++){
                xtilde[index3D(i, j, k)] /= (t1 + t2 + compute_eigenvalue_neumann(k, N) )/std::pow(h, 2);
            }
        }
    }
    xtilde[0] = 0;

    // inverse transform
    fftw_plan xtilde_to_x_plan= fftw_plan_r2r_1d(N, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);
    for (int kk = 0; kk < 3; kk++){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++){
                extract_array(xtilde, N, i*N*N + j*N, temp1);

                fftw_execute(xtilde_to_x_plan);
                for (int k = 0; k < N; k++){
                    x[index3D(i, j, k)] = ( temp2[k] )/( 2.0 * (N-1) ); // 
                }
            }
        }
        c2d->transposeX2Y_MajorIndex(x, x);
        xtilde = x;
    }

    double difference = x[0] - Uex[0];
    double mazx = -2.0;
    for(int i=0; i<size; ++i)
    {
        // assert(x[i] - Uex[i] -difference  <= 1e-15 );
        double temp = std::abs(x[i] - Uex[i] - difference );
        if (mazx < temp) mazx = temp;
    }
    cout<< mazx << endl;

    //Now lets kill MPI
    fftw_destroy_plan(b_to_btilde_plan);
    fftw_destroy_plan(xtilde_to_x_plan);
    fftw_cleanup();
    MPI_Finalize();

    return 0;
}

