//TODO 
// 
// OK: Initialize the rhs in order to match the domain decomp
// OK: Define size local for each pencil
// OK: Understand how to implement in order to avoid the x direction to be the slowest (Won't need exxtract array?)
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
    //cout << mpiRank << " " << totRank;

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
    // As thought pencils are 4*2*2 (x, y, z)
    //cout <<"Nzloc = " << NzLoc << "\n";
    //cout <<"Nyloc = " << NyLoc << "\n";
    //cout <<"Nxloc = " << NxLoc << "\n";
    auto index3dLocal =[&](int i, int j, int k) {
        return (k * NxLoc * NyLoc) + (j * NxLoc) + i;
    };
    int Globsize = Nx * Ny * Nz;
    int size = NxLoc * NyLoc * NzLoc;

    double *Uex   ;
    double *b     ;
    // Vectors with local informations
    c2d->allocX( Uex );
    c2d->allocX( b );



    // Create forcing term manufactured from manufactured sulotion (u =-3cx*cy*cz, x, y, z = [0, 2pi])
    double hx = 2*PI/(Nx-1) ;
    double hy = 2*PI/(Ny-1) ;
    double hz = 2*PI/(Nz-1) ;

    //Riempimento
    for (int i = 0; i <= c2d->xEnd[0]; i++) {
        for (int j = 0; j <= c2d->xEnd[1]; j++) {
            for (int k = 0; k <= c2d->xEnd[2]; k++) {
                // Need to fill the LOCAL Vector with GLOBAL values !!
                int jGlobal = j + c2d->xStart[1], kGlobal = k+c2d->xStart[2];
                Uex[index3dLocal(i, j, k)] = std::cos(hx*i) * std::cos(hy*jGlobal) * std::cos(hz*kGlobal) ;
                b[index3dLocal(i, j, k)] = -3.0*Uex[index3dLocal(i, j, k)];
            }    
        }
    }

    /* CHECK RIEMPIMENTO, ingestibile con mpi di mezzo dc
  //// this element end is in (3, 3, 1)
  //cout << "x0 = " << c2d->xEnd[0] << "\n"; 
  //cout << "y0 = " << c2d->xEnd[1] << "\n"; 
  //cout << "z0 = " << c2d->xEnd[2] << "\n"; 
  //// this element start is in (0, 2, 0)
  //cout << "x0 = " << c2d->xStart[0] << "\n"; 
  //cout << "y0 = " << c2d->xStart[1] << "\n"; 
  //cout << "z0 = " << c2d->xStart[2] << "\n"; 

    for (int k = 0; k <= c2d->xEnd[2]; k++) {
        cout << " New Plane \n";
        for (int j = 0; j <= c2d->xEnd[1]; j++) {
            cout << "\n";
            for (int i = 0; i <= c2d->xEnd[0]; i++) {
                cout << 
                    Uex[index3dLocal(i, j, k)] << "|" 
                    << (k * Nx * Ny) + (j * Nx) + i << " \t" ;
            }    
        }
    }
    */

    double *x      = (double*) fftw_malloc(sizeof(double) * size);
    //  Not needed, we do them in place
    double *btilde = (double*) fftw_malloc(sizeof(double) * size);	
    double *xtilde = (double*) fftw_malloc(sizeof(double) * size);


    double *temp1 = (double*) fftw_malloc(sizeof(double) * N);
    double *temp2 = (double*) fftw_malloc(sizeof(double) * N);
    // Exucute dct type 1 along all 3 directions
    fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(N, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    /* OLD VERSION !!
    for (int kk = 0; kk < 3; kk++){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++){
                extract_array(b, N, i*N*N + j*N, temp1);
                fftw_execute(b_to_btilde_plan);
                for (int k = 0; k < N; k++){
                    btilde[index3dLocal(i, j, k)] = temp2[k];
                }
            }
        }
        // TODO Funziona in parallelo che itero la rotazione xtoy??
        c2d->transposeX2Y_MajorIndex(btilde, btilde);

        b = btilde;
    }
    // Rename btilde
    xtilde = btilde;

    double h = 1.0;
    for (int i = 0; i < N; i++) {
        // double t1= compute_eigenvalue_neumann(i, N);
        for (int j = 0; j < N; j++){
            double t2= compute_eigenvalue_neumann(j, N);
            for (int k = 0; k < N; k++){
                xtilde[index3dLocal(i, j, k)] /= (t1 + t2 + compute_eigenvalue_neumann(k, N) )/std::pow(h, 2);
            }
        }
    }
    */
    /* OLD VERSION !!
    // inverse transform
    fftw_plan xtilde_to_x_plan= fftw_plan_r2r_1d(N, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);
    for (int kk = 0; kk < 3; kk++){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++){
                extract_array(xtilde, N, i*N*N + j*N, temp1);

                fftw_execute(xtilde_to_x_plan);
                for (int k = 0; k < N; k++){
                    x[index3dLocal(i, j, k)] = ( temp2[k] )/( 2.0 * (N-1) ); // 
                }
            }
        }
        c2d->transposeX2Y_MajorIndex(x, x);
        xtilde = x;
    }
*/

    // DCT along x
    for (int j = 0; j < NyLoc; j++) {
        for (int k = 0; k < NzLoc; k++) {
            // Access to the subarry element
            // temp1[:] = rhs(:, j, k)
            for (int i = 0; i < NxLoc; i++) {
                temp1[i] = b[index3dLocal(i, j, k)];
            }
            // Actual fft (from temp1 to temp2)
            fftw_execute(b_to_btilde_plan);
            // Store in btilde
            for (int i = 0; i < NxLoc; i++) {
                btilde[index3dLocal(i, j, k)] = temp2[i];
            }
        }
    }
    // Transpose xtoy and rename NiLocs
    c2d->transposeX2Y(btilde, btilde);
    NxLoc = c2d->ySize[0];
    NyLoc = c2d->ySize[1];
    NzLoc = c2d->ySize[2];
    // dct along y
    for (int i = 0; i < NxLoc; i++) {
        for (int k = 0; k < NzLoc; k++) {
            // Access to the subarry element
            // temp1[:] = rhs(i, :, k)
            for (int j = 0; j < NyLoc; j++) {
                temp1[j] = b[index3dLocal(i, j, k)];
            }
            // Actual fft (from temp1 to temp2)
            fftw_execute(b_to_btilde_plan);
            // Store in btilde
            for (int j = 0; j < NyLoc; j++) {
                btilde[index3dLocal(i, j, k)] = temp2[j];
            }
        }
    }
    // Transpose ytoz and rename NiLocs
    c2d->transposeY2Z(btilde, btilde);
    NxLoc = c2d->zSize[0];
    NyLoc = c2d->zSize[1];
    NzLoc = c2d->zSize[2];

    // dct along z
    for (int i = 0; i < NxLoc; i++) {
        for (int j = 0; j < NyLoc; j++) {
            // Access to the subarry element
            // temp1[:] = rhs(i, :, k)
            for (int k = 0; k < NzLoc; k++) {
                temp1[k] = b[index3dLocal(i, j, k)];
            }
            // Actual fft (from temp1 to temp2)
            fftw_execute(b_to_btilde_plan);
            // Store in btilde
            for (int k = 0; k < NzLoc; k++) {
                btilde[index3dLocal(i, j, k)] = temp2[k];
            }
        }
    }

    // Rename btilde
    xtilde = btilde;


    // SOLVE FOR Xtilda  && Compute eigenvalues
    // !! Datas are still stored in Z mayor axes !!
    for (int i = 0; i < NxLoc; i++) {
        int iGlob = c2d->zStart[0] + i;
        double t1 = (2.0 *cos( PI * iGlob/ (N-1)) - 2.0)/(hx*hx);
        for (int j = 0; j < NyLoc; j++) {
            int jGlob = c2d->zStart[1] + j;
            double t2 = (2.0 *cos( PI * jGlob/ (N-1)) - 2.0)/(hy*hy);
            for (int k = 0; k < NzLoc; k++) {
                int kGlob = c2d->zStart[2] + k;
                double t3 = (2.0 *cos( PI * kGlob/ (N-1)) - 2.0)/(hz*hz);
                xtilde[index3dLocal(i, j, k)] /= (t1 + t2 + t3); 
            }
        }
    }

    // Set xtilda( 3dGLOBAL(0,  0, 0) = 0 )
    if (!mpiRank) {
        xtilde[0] = 0;
    }

    fftw_plan xtilde_to_x_plan= fftw_plan_r2r_1d(N, temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);
    // !! Datas are still stored in Z mayor axes !!
    // idct along z
    for (int i = 0; i < NxLoc; i++) {
        for (int j = 0; j < NyLoc; j++) {
            // Access to the subarry element
            // temp1[:] = xtilde(i, :, k)
            for (int k = 0; k < NzLoc; k++) {
                temp1[k] = xtilde[index3dLocal(i, j, k)];
            }
            // Actual fft (from temp1 to temp2)
            fftw_execute(xtilde_to_x_plan);
            // Store in btilde
            for (int k = 0; k < NzLoc; k++) {
                btilde[index3dLocal(i, j, k)] = temp2[k];
            }
        }
    }
    // transp ztoy && rename NiLoc
    c2d->transposeZ2Y(xtilde, xtilde);
    NxLoc = c2d->ySize[0];
    NyLoc = c2d->ySize[1];
    NzLoc = c2d->ySize[2];

    // idct along y
    for (int i = 0; i < NxLoc; i++) {
        for (int k = 0; k < NzLoc; k++) {
            // Access to the subarry element
            // temp1[:] = xtilde(i, :, k)
            for (int j = 0; j < NyLoc; j++) {
                temp1[j] = xtilde[index3dLocal(i, j, k)];
            }
            // Actual fft (from temp1 to temp2)
            fftw_execute(xtilde_to_x_plan);
            // Store in btilde
            for (int j = 0; j < NyLoc; j++) {
                btilde[index3dLocal(i, j, k)] = temp2[k];
            }
        }
    }
    // transp ytox
    c2d->transposeY2X(xtilde, xtilde);
    NxLoc = c2d->xSize[0];
    NyLoc = c2d->xSize[1];
    NzLoc = c2d->xSize[2];
    // !! Datas are NOW stored in X mayor axes !!

    // idct along x
    for (int j = 0; j < NyLoc; j++) {
        for (int k = 0; k < NzLoc; k++) {
            // Access to the subarry element
            // temp1[:] = xtilde(i, :, k)
            for (int i = 0; i < NxLoc; i++) {
                temp1[i] = xtilde[index3dLocal(i, j, k)];
            }
            // Actual fft (from temp1 to temp2)
            fftw_execute(xtilde_to_x_plan);
            // Store in btilde
            for (int i = 0; i < NxLoc; i++) {
                btilde[index3dLocal(i, j, k)] = temp2[i];
            }
        }
    }

    // NOW THE SOLUTION IS STORED IN EACH PROCESSOR, in the rel solver the work finisces here, but we need to CHECK so: 
    // - Get errmax in each processor, (store in a Vector)
    // - Store it in a vector
    // - Check the max and finally shoud be it

    // BLOCK ALL PROCS AT THIS POINT
    int MPI_Barrier( MPI_Comm comm );
    double difference = 0;
    double diffGlob;
    if (mpiRank==0) {
        // Of course only rank 0 has x(0) and Uex(0) GLOBALS
        difference = x[0] - Uex[0];
    }
    diffGlob = difference;
    int MPI_Barrier( MPI_Comm comm );
    //  GET MAX in each processor, CAN'T DO  IT PROPERLY, seems like it  takes values from random points in memory...
    double mazx = -2.0;
    for(int i=0; i<size; ++i)
    {
        // assert(x[i] - Uex[i] -difference  <= 1e-15 );
        double temp = std::abs(x[i] - Uex[i] - diffGlob );
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

