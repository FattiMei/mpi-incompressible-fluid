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

    int N = 4;
    // int size = N*N*N;

    // double *b      = (double*) fftw_malloc(sizeof(double) * size);

    // double h = 2*PI/(N-1) ;
    // for (int i = 0; i < Nx; i++) {
    //     for (int j = 0; j < Ny; j++) {
    //         for (int k = 0; k < Nz; k++) {
    //             Uex[index3D(i,  j,  k,  N)] = std::cos(h*i) * std::cos(h*j) * std::cos(h*k) ;
    //             b[index3D(i, j, k, N)] = -3.0*Uex[index3D(i,  j,  k,  N)];
    //         }    
    //     }
    // }

    if(!mpiRank){
        cout << endl;
        cout << "-------------------" << endl;
    	cout << " C2Decomp Testing " << endl;
    	cout << "-------------------" << endl;
    	cout << endl;
    }


	
    //MPI_Barrier(MPI_COMM_WORLD);
    // Create forcing term manufactured from manufactured sulotion (u =-3cx*cy*cz, x, y, z = [0, 2pi])
    
    

    // pRow*pCol = rankMPI to work properly!!

    

    bool neumannBC[3] = {true, true, true};

    if(!mpiRank) cout << "initializing " << endl;

    C2Decomp *c2d = new C2Decomp(N, N, N, 0, 0, neumannBC);

    if(!mpiRank) cout << "done initializing " << endl;

    int xSize[3] = {c2d->xSize[0], c2d->xSize[1], c2d->xSize[2]};
    int ySize[3] = {c2d->ySize[0], c2d->ySize[1], c2d->ySize[2]};
    int zSize[3] = {c2d->zSize[0], c2d->zSize[1], c2d->zSize[2]};

    double *temp1=NULL, *temp2=NULL;

    double *u1, *u2, *u3, *div1, *div2, *div3;

    double *Uex    = (double*) fftw_malloc(sizeof(double) * xSize[2] * xSize[1] * xSize[0]);

    c2d->allocX(u1);
    c2d->allocY(u2);
    c2d->allocZ(u3);
    c2d->allocX(div1);
    c2d->allocY(div2);
    c2d->allocZ(div3);


    double h = 2*PI/(N-1) ;

    for (int i = 0; i < xSize[2]; i++) {
        for (int j = 0; j < xSize[1]; j++){
            for (int k = 0; k < xSize[0]; k++){
                //div1[i*xSize[1]*xSize[0] + j*xSize[0] + k] = (i + c2d->xStart[2])*xSize[1]*xSize[0] + (j + c2d->xStart[1])*xSize[0] + (k + c2d->xStart[0]);
                Uex[i*xSize[1]*xSize[0] + j*xSize[0] + k] = std::cos(h*(i + c2d->xStart[2])) * std::cos(h*(j + c2d->xStart[1])) * std::cos(h*(k + c2d->xStart[0])) ;
                div1[i*xSize[1]*xSize[0] + j*xSize[0] + k] = -3.0*Uex[i*xSize[1]*xSize[0] + j*xSize[0] + k];// i*xSize[1]*xSize[0] + j*xSize[0] + k + 1;
                // cout<< div1[i*xSize[1]*xSize[0] + j*xSize[0] + k] << " ("<< mpiRank << ") ";
            }
        }
    }
    //cout<<endl;
    

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

    c2d->transposeX2Y_MajorIndex(u1, u2);
    c2d->transposeX2Y_MajorIndex(div1, div2);

    

    temp1 = new double[ySize[0]];
    temp2 = new double[ySize[0]];

    b_to_btilde_plan = fftw_plan_r2r_1d(ySize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < ySize[2]; i++) {
        for (int j = 0; j < ySize[1]; j++){
            extract_array(u2, ySize[0], i*ySize[1]*ySize[0] + j*ySize[0], temp1);
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
            extract_array(u3, zSize[0], i*zSize[1]*zSize[0] + j*zSize[0], temp1);
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

    // finish 1 time transform

    for (int i = 0; i < xSize[2]; i++) {
        double t1= compute_eigenvalue_neumann(i, N);
        for (int j = 0; j < xSize[1]; j++){
            double t2= compute_eigenvalue_neumann(j, N);
            for (int k =0; k < xSize[0]; k++){
                u1[i*xSize[1]*xSize[0] + j*xSize[0] + k] /= (t1 + t2 + compute_eigenvalue_neumann(k, N) )/std::pow(h, 2);
            }
        }
    }

    u1[c2d->xStart[0]] = 0;

    // Init anti transform

    temp1 = new double[xSize[0]];
    temp2 = new double[xSize[0]];

    b_to_btilde_plan = fftw_plan_r2r_1d(xSize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < xSize[2]; i++) {
        for (int j = 0; j < xSize[1]; j++){
            extract_array(u1, xSize[0], i*xSize[1]*xSize[0] + j*xSize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < xSize[0]; k++){
                u1[i*xSize[1]*xSize[0] + j*xSize[0] + k] = ( temp2[k] )/( 2.0 * (N-1) );
            }
        }
    }

    free(temp1);
    free(temp2);

    c2d->transposeX2Y_MajorIndex(u1, u2);
    c2d->transposeX2Y_MajorIndex(div1, div2);

    

    temp1 = new double[ySize[0]];
    temp2 = new double[ySize[0]];

    b_to_btilde_plan = fftw_plan_r2r_1d(ySize[0], temp1, temp2, FFTW_REDFT00, FFTW_ESTIMATE);

    for (int i = 0; i < ySize[2]; i++) {
        for (int j = 0; j < ySize[1]; j++){
            extract_array(u2, ySize[0], i*ySize[1]*ySize[0] + j*ySize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < ySize[0]; k++){
                u2[i*ySize[1]*ySize[0] + j*ySize[0] + k] = ( temp2[k] )/( 2.0 * (N-1) );
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
            extract_array(u3, zSize[0], i*zSize[1]*zSize[0] + j*zSize[0], temp1);
            fftw_execute(b_to_btilde_plan);
            for (int k = 0; k < zSize[0]; k++){
                u3[i*zSize[1]*zSize[0] + j*zSize[0] + k] = ( temp2[k] )/( 2.0 * (N-1) );
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

    // double difff = u1[c2d->xStart[0]] - Uex[c2d->xStart[0]];

    // double max = 0.0;


    // for (int i = c2d->xStart[2]; i < xSize[2]; i++) {
    //     for (int j = c2d->xStart[1]; j < xSize[1]; j++){
    //         for (int k = c2d->xStart[0]; k < xSize[0]; k++){
    //             //cout<< div1[i*xSize[1]*xSize[0] + j*xSize[0] + k] << " ";
    //             double temppp = std::abs(u1[i*xSize[1]*xSize[0] + j*xSize[0] + k] - Uex[i*xSize[1]*xSize[0] + j*xSize[0] + k] - difff);
    //             if (max < temppp)
    //                 max = temppp;
    //         }
    //     }
    // }
    
    // cout <<endl<< "And the max error from processor " <<  mpiRank << " is "<< max <<endl << endl;

    std::vector<double> gathered_results;

    if (mpiRank == 0) {
        gathered_results.resize(N*N*N);
    }

    MPI_Gather(
        u1,               // Address of the data to send
        xSize[2]*xSize[1]*xSize[0],                 // Number of elements to send (1 int here)
        MPI_DOUBLE,           // Data type of the element
        gathered_results.data(), // Address of the receive buffer on root (only relevant on rank 0)
        xSize[2]*xSize[1]*xSize[0],                 // Number of elements each process contributes
        MPI_DOUBLE,           // Data type of the received elements
        0,                 // Rank of the root process
        MPI_COMM_WORLD     // Communicator
    );

    if (mpiRank == 0) {
        std::cout << "Gathered results at rank 0: ";
        for (const auto& val : gathered_results) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    for (int iii = 0; iii < totRank; iii++){
        if(mpiRank == iii){
            cout<< "From Processor " << mpiRank <<endl;
            double difff = u1[c2d->xStart[0]] - Uex[c2d->xStart[0]];

            double max = 0.0;

            for (int i = 0; i < xSize[2]; i++) {
                for (int j = 0; j < xSize[1]; j++){
                    for (int k = 0; k < xSize[0]; k++){
                        //cout<< div1[i*xSize[1]*xSize[0] + j*xSize[0] + k] << " ";
                        double temppp = std::abs(u1[i*xSize[1]*xSize[0] + j*xSize[0] + k] - Uex[i*xSize[1]*xSize[0] + j*xSize[0] + k] - difff);
                        if (max < temppp)
                            max = temppp;
                    }
                }
            }

            cout <<endl<< "And the max error is "<< max <<endl << endl;

            for(int i = 0; i < 3; i++){
                cout << xSize[i] << " "<< ySize[i] << " "<< zSize[i] << endl;
                cout << "My starting x point: "<< c2d->xStart[2] << " " << c2d->xStart[1] << " " << c2d->xStart[0] << endl;
                cout << "My starting y point: "<< c2d->yStart[2] << " " << c2d->yStart[1] << " " << c2d->yStart[0] << endl;
                cout << "My starting z point: "<< c2d->zStart[2] << " " << c2d->zStart[1] << " " << c2d->zStart[0] << endl;
            }
            cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    fftw_free(Uex);
    fftw_destroy_plan(b_to_btilde_plan);

    c2d->deallocXYZ(u1);
    c2d->deallocXYZ(div1);
    c2d->deallocXYZ(u2);
    c2d->deallocXYZ(div2);
    c2d->deallocXYZ(u3);
    c2d->deallocXYZ(div3);

    fftw_cleanup();
    MPI_Finalize();

	return 0;
}