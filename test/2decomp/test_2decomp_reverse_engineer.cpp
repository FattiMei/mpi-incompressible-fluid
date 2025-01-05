#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <mpi.h>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop


using namespace std;


inline int index3D(int i, int j, int k, int N) {
	return (i * N * N) + (j * N) + k;
}


int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: <program> <number of nodes>" << std::endl;
		exit(1);
	}

	// the computational mesh is a cube for now, but we will need general dimensions
	const int N = std::atoi(argv[1]);
	const int Nx = N,
	          Ny = N,
		  Nz = N;
	const int size = Nx * Ny * Nz;


	// mpi initialization, the rank and size will be parameters of 2Decomp
	int ierr;
	int totRank, mpiRank;

	ierr = MPI_Init( &argc, &argv);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);


	// someone has found that this condition is necessary for the program, otherwise 2Decomp crashes
	// leave 0,0 for autotuning (SLOW)
	const int pRow = 2, pCol = 2;
	assert(pRow * pCol == totRank);


	// no required boundary conditions for this program
	bool neumannBC[3] = {false, false, false};
	C2Decomp *c2d = new C2Decomp(N, N, N, pRow, pCol, neumannBC);


	// investigate the domain decomposition, starting from the C2Decomp::xStart...
	if (!mpiRank) {
		std::cout << "I'm the first processor, asking about decomposition" << std::endl;

		std::cout
			<< "xStart (x y z): "
			<< c2d->xStart[0]
			<< ' '
			<< c2d->xStart[1]
			<< ' '
			<< c2d->xStart[2]
			<< std::endl
			<< "xEnd (x y z): "
			<< c2d->xEnd[0]
			<< ' '
			<< c2d->xEnd[1]
			<< ' '
			<< c2d->xEnd[2]
			<< std::endl
			<< "yStart (y y z): "
			<< c2d->yStart[0]
			<< ' '
			<< c2d->yStart[1]
			<< ' '
			<< c2d->yStart[2]
			<< std::endl
			<< "yEnd (y y z): "
			<< c2d->yEnd[0]
			<< ' '
			<< c2d->yEnd[1]
			<< ' '
			<< c2d->yEnd[2]
			<< std::endl;

		std::cout
			<< "plane size: " << c2d->xSize[1] * c2d->xSize[2] << std::endl
			<< "line size: " << c2d->xSize[1] << std::endl;
	}

	Real *reference_domain_X = NULL;
	Real *alternate_domain_X = NULL;
	Real *local_domain_Y = NULL;

	const int ncells = c2d->allocX(reference_domain_X);
	c2d->allocX(alternate_domain_X);
	c2d->allocY(local_domain_Y);

	for (int i = c2d->xStart[0]; i < c2d->xEnd[0]; ++i) {
		for (int j = c2d->xStart[1]; j < c2d->xEnd[1]; ++j) {
			for (int k = c2d->xStart[2]; k < c2d->xEnd[2]; ++k) {
				const int plane_size = c2d->xSize[1] * c2d->xSize[2];
				const int line_size = c2d->xSize[1];

				reference_domain_X[i*plane_size + j*line_size + k] = i*N*N + j*N + k;
			}
		}
	}

	for (int i = 0; i < ncells; ++i) {
		alternate_domain_X[i] = reference_domain_X[i];
	}

	// original, works
	c2d->transposeX2Y(reference_domain_X, local_domain_Y);
	c2d->transposeY2X(local_domain_Y, alternate_domain_X);

	// in place transpositions (THEY ARE NOT CORRECT! FUCK CHATGPT)
	// c2d->transposeX2Y(alternate_domain_X, alternate_domain_X);
	// c2d->transposeX2Y(alternate_domain_X, alternate_domain_X);

	if (true) {
		for (int i = c2d->xStart[0]; i < c2d->xEnd[0]; ++i) {
			for (int j = c2d->xStart[1]; j < c2d->xEnd[1]; ++j) {
				for (int k = c2d->xStart[2]; k < c2d->xEnd[2]; ++k) {
					const int plane_size = c2d->xSize[1] * c2d->xSize[2];
					const int line_size = c2d->xSize[1];

					const bool ok =
						reference_domain_X[i*plane_size + j*line_size + k]
						==
						alternate_domain_X[i*plane_size + j*line_size + k];

					if (not ok) {
						std::cout << "Errore nella trasposizione" << std::endl;
					}
				}
			}
		}
	}


	//Now lets kill MPI
	MPI_Finalize();


	return 0;
}
