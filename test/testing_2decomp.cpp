#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <fftw3.h>
#include <mpi.h>
#include <stdlib.h>
#include "../deps/2Decomp_C/C2Decomp.hpp"

constexpr double PI = 3.141592653589793;

using namespace std;

inline double compute_eigenvalue_neumann(int index, int N) {
	return (2.0 *cos( PI * index / (N-1)) - 2.0);
}

inline int index3D(int i, int j, int k, int N) {
    return (i * N * N) + (j * N) + k;
}

void extract_array(double arr[], int size, int start, double subarray[]){
    // Copy the elements manually
    for (int i = 0; i < size; i++) {
        subarray[i] = arr[start + i];
    }
}

int main(int argc, char *argv[]) {

    int ierr, totRank, mpiRank;

    ierr = MPI_Init( &argc, &argv);

    //Get the number of processes
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);

    //Get the local rank
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    int N = 10;
    int size = N*N*N;

    double *Uex    = (double*) fftw_malloc(sizeof(double) * size);

    //double *b      = (double*) fftw_malloc(sizeof(double) * size);

    //double *xFinal    = (double*) fftw_malloc(sizeof(double) * size);

    if(!mpiRank){
        cout << endl;
        cout << "-------------------" << endl;
    	cout << " C2Decomp Testing " << endl;
    	cout << "-------------------" << endl;
    	cout << endl;

        
        try
        {
            if (argc > 1) {
                N = std::stoi(argv[1]); // Use std::stoi for better error detection
                cout << "N set equal to " << N << endl <<endl;
            } else {
                cout << "No argument provided. N set to default value: 3" << endl << endl;
            }
        }catch (const std::invalid_argument& e) {
            cout << "N set equal to 3"<<endl<<endl;
        }catch (...) {
            std::cerr << "Unknown error occurred. N set to default value: 3" << std::endl << std::endl;
        }

        int Nx, Ny, Nz;
        Nx = N;
        Ny = N;
        Nz = N;

        size = Nx * Ny * Nz;

        double h = 2*PI/(N-1) ;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    Uex[index3D(i,  j,  k,  N)] = std::cos(h*i) * std::cos(h*j) * std::cos(h*k) ;
                    //b[index3D(i, j, k, N)] = -3.0*Uex[index3D(i,  j,  k,  N)];
                }    
            }
        }

    }

    double *b11      = new double[N * N * N];
    double *b12      = new double[N * N * N];

    for (int i = 0; i < size; i++){
        b11[i] = i+1;
    }
	
    MPI_Barrier(MPI_COMM_WORLD);
    // Create forcing term manufactured from manufactured sulotion (u =-3cx*cy*cz, x, y, z = [0, 2pi])
    
    

    // pRow*pCol = rankMPI to work properly!!

    int pCol = 0;
    int pRow = 0;

    bool neumannBC[3] = {true, true, true};

    if(!mpiRank) cout << "initializing " << endl;
    C2Decomp *c2d = new C2Decomp(N, N, N, 0, 0, neumannBC);
    
    if(!mpiRank) cout << "done initializing " << endl;

    int xSize[3] = {c2d->xSize[0], c2d->xSize[1], c2d->xSize[2]};
    int ySize[3] = {c2d->ySize[0], c2d->ySize[1], c2d->ySize[2]};
    int zSize[3] = {c2d->zSize[0], c2d->zSize[1], c2d->zSize[2]};

    double *temp1=NULL, *temp2=NULL;

    double *u1, *u2, *u3, *div1, *div2, *div3;



    c2d->allocX(u1);
    c2d->allocY(u2);
    c2d->allocZ(u3);
    c2d->allocX(div1);
    c2d->allocY(div2);
    c2d->allocZ(div3);

    for(int ip = 0; ip < xSize[2]*xSize[1]*xSize[0]; ip++){
	    div1[ip] = ip + 1;
    }

    temp1 = new double[xSize[0]];
    temp2 = new double[xSize[0]];

    fftw_plan b_to_btilde_plan = fftw_plan_r2r_1d(xSize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < xSize[2]; i++) {
        for (int j = 0; j < xSize[1]; j++){
            extract_array(div1, xSize[0], i*xSize[1]*xSize[0] + j*xSize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < xSize[0]; k++){
                u1[i*xSize[1]*xSize[0] + j*xSize[0] + k] = temp2[k];
            }
        }
    }

    free(temp1);
    free(temp2);

    c2d->transposeX2Y(u1, u2);
    c2d->transposeX2Y(div1, div2);

    

    temp1 = new double[ySize[0]];
    temp2 = new double[ySize[0]];

    b_to_btilde_plan = fftw_plan_r2r_1d(ySize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < ySize[2]; i++) {
        for (int j = 0; j < ySize[1]; j++){
            extract_array(div2, ySize[0], i*ySize[1]*ySize[0] + j*ySize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < ySize[0]; k++){
                u2[i*ySize[1]*ySize[0] + j*ySize[0] + k] = temp2[k];
            }
        }
    }

    free(temp1);
    free(temp2);

    c2d->transposeY2Z_MajorIndex(u2, u3);
    c2d->transposeY2Z_MajorIndex(div2, div3);

    temp1 = new double[zSize[0]];
    temp2 = new double[zSize[0]];


    b_to_btilde_plan = fftw_plan_r2r_1d(zSize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < zSize[2]; i++) {
        for (int j = 0; j < zSize[1]; j++){
            extract_array(div3, zSize[0], i*zSize[1]*zSize[0] + j*zSize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < zSize[0]; k++){
                u3[i*zSize[1]*zSize[0] + j*zSize[0] + k] = temp2[k];
            }
        }
    }

    free(temp1);
    free(temp2);

    c2d->transposeZ2Y_MajorIndex(u3, u2);
    c2d->transposeZ2Y_MajorIndex(div3, div2);
    c2d->transposeY2X_MajorIndex(u2, u1);
    c2d->transposeY2X_MajorIndex(div2, div1);


    MPI_Barrier(MPI_COMM_WORLD);

    // finish 1 time transform

    temp1 = new double[xSize[0]];
    temp2 = new double[xSize[0]];

    b_to_btilde_plan = fftw_plan_r2r_1d(xSize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < xSize[2]; i++) {
        for (int j = 0; j < xSize[1]; j++){
            extract_array(div1, xSize[0], i*xSize[1]*xSize[0] + j*xSize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < xSize[0]; k++){
                u1[i*xSize[1]*xSize[0] + j*xSize[0] + k] = temp2[k];
            }
        }
    }

    free(temp1);
    free(temp2);

    c2d->transposeX2Y(u1, u2);
    c2d->transposeX2Y(div1, div2);

    

    temp1 = new double[ySize[0]];
    temp2 = new double[ySize[0]];

    b_to_btilde_plan = fftw_plan_r2r_1d(ySize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < ySize[2]; i++) {
        for (int j = 0; j < ySize[1]; j++){
            extract_array(div2, ySize[0], i*ySize[1]*ySize[0] + j*ySize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < ySize[0]; k++){
                u2[i*ySize[1]*ySize[0] + j*ySize[0] + k] = temp2[k];
            }
        }
    }

    free(temp1);
    free(temp2);

    c2d->transposeY2Z_MajorIndex(u2, u3);
    c2d->transposeY2Z_MajorIndex(div2, div3);

    temp1 = new double[zSize[0]];
    temp2 = new double[zSize[0]];


    b_to_btilde_plan = fftw_plan_r2r_1d(zSize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < zSize[2]; i++) {
        for (int j = 0; j < zSize[1]; j++){
            extract_array(div3, zSize[0], i*zSize[1]*zSize[0] + j*zSize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < zSize[0]; k++){
                u3[i*zSize[1]*zSize[0] + j*zSize[0] + k] = temp2[k];
            }
        }
    }

    free(temp1);
    free(temp2);

    c2d->transposeZ2Y_MajorIndex(u3, u2);
    c2d->transposeZ2Y_MajorIndex(div3, div2);
    c2d->transposeY2X_MajorIndex(u2, u1);
    c2d->transposeY2X_MajorIndex(div2, div1);

    

    for (int i = 0; i < totRank; i++){
        if(mpiRank == i){
            cout<< "From Processor " << mpiRank <<endl;
            double difff = div1[0] - 1;
            for (int i = 0; i < xSize[2] * xSize[1] * xSize[0]; i++){
                cout << std::abs(div1[i] +1 - i - difff) << " ";
            }
            cout<<endl;
            for(int i = 0; i < 3; i++){
                cout << xSize[i] << " "<< ySize[i] << " "<< zSize[i] << endl;
            }
            cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    fftw_free(Uex);
    fftw_free(b11);
    fftw_free(b12);

    fftw_cleanup();
    MPI_Finalize();

	return 0;
}