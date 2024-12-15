#include <iostream>
#include "StaggeredTensor.h"
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop

// A simple test to verify how 2decomp operates on tensors.
int main(int argc, char *argv[]) {
    int rank;
    int size;

    // Initialize the MPI environment.
    MPI_Init(&argc, &argv);
    
    // Get the rank of the current processor.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processors.
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using namespace mif;

    // Create a Nx x Ny x Nz tensor.
    const size_t Nx = 4;
    const size_t Ny = 3;
    const size_t Nz = 2;
    const Constants constants(Nx, Ny, Nz, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 1, 1, 0);
    StaggeredTensor tensor({Nx, Ny, Nz}, constants);

    // Initialize the tensor with easily recognizable values.
    for (size_t k = 0; k < Nz; k++) {
        for (size_t j = 0; j < Ny; j++) {
            for (size_t i = 0; i < Nx; i++) {
                tensor(i,j,k) = i + 10*j + k*100;
            }
        }
    }

    std::cout << "Original tensor: " << std::endl;
    for (size_t i = 0; i < Nx*Ny*Nz; i++) {
        std::cout << (static_cast<Real*>(tensor.raw_data()))[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides: x=1, y=Nx, z=Nx*Ny" << std::endl;

    // Create 2decomp objects.
    bool neumannBC[3] = {true, true, true};
    C2Decomp c2d = C2Decomp(Nx, Ny, Nz, 1, 1, neumannBC);

    // Transpose.
    c2d.transposeX2Y_MajorIndex(static_cast<Real*>(tensor.raw_data()), static_cast<Real*>(tensor.raw_data()));
    std::cout << "Transposed tensor X2Y: " << std::endl;
    for (size_t i = 0; i < Nx*Ny*Nz; i++) {
        std::cout << (static_cast<Real*>(tensor.raw_data()))[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides: x=Ny*Nz, y=1, z=Ny" << std::endl;

    // Transpose again.
    c2d.transposeY2Z_MajorIndex(static_cast<Real*>(tensor.raw_data()), static_cast<Real*>(tensor.raw_data()));
    std::cout << "Transposed tensor Y2Z: " << std::endl;
    for (size_t i = 0; i < Nx*Ny*Nz; i++) {
        std::cout << (static_cast<Real*>(tensor.raw_data()))[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides: x=Nz, y=Nx*Nz, z=1" << std::endl;
    
    // Transpose back.
    c2d.transposeZ2Y_MajorIndex(static_cast<Real*>(tensor.raw_data()), static_cast<Real*>(tensor.raw_data()));
    c2d.transposeY2X_MajorIndex(static_cast<Real*>(tensor.raw_data()), static_cast<Real*>(tensor.raw_data()));
    std::cout << "Copy of original tensor: " << std::endl;
    for (size_t i = 0; i < Nx*Ny*Nz; i++) {
        std::cout << (static_cast<Real*>(tensor.raw_data()))[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides: x=1, y=Nx, z=Nx*Ny" << std::endl;

    MPI_Finalize();
    return 0;
}