#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "../deps/2Decomp_C/C2Decomp.hpp"
#pragma GCC diagnostic pop
#include "StaggeredTensor.h"
#include "VelocityTensor.h"

template <typename T> void print_t(T &arr, int n1, int n2, int n3) {
  std::cout << "To visualize: https://array-3d-viz.vercel.app/\n";
  std::cout << "[\n";
  for (int kp = 0; kp < n1; kp++) {
    std::cout << "[";
    for (int jp = 0; jp < n2; jp++) {
      std::cout << "[";
      for (int ip = 0; ip < n3; ip++) {
        if (ip == n3 - 1) {
          std::cout << arr(ip, jp, kp);
        } else {
          std::cout << arr(ip, jp, kp) << ',';
        }
      }
      if (jp == n2 - 1) {
        std::cout << "]";

      } else {
        std::cout << "],";
      }
    }
    if (kp == n1 - 1) {
      std::cout << "]\n";
    } else {
      std::cout << "],\n";
    }
  }
  std::cout << "]\n";
}

template <typename T> void print_pointer(T *arr, int n1, int n2, int n3) {
  std::cout << "To visualize: https://array-3d-viz.vercel.app/\n";
  std::cout << "[\n";
  for (int kp = 0; kp < n1; kp++) {
    std::cout << "[";
    for (int jp = 0; jp < n2; jp++) {
      std::cout << "[";
      for (int ip = 0; ip < n3; ip++) {
        const int index = kp * (n2 * n3) + jp * n3 + ip;
        if (ip == n3 - 1) {
          std::cout << arr[index];
        } else {
          std::cout << arr[index] << ',';
        }
      }
      if (jp == n2 - 1) {
        std::cout << "]";

      } else {
        std::cout << "],";
      }
    }
    if (kp == n1 - 1) {
      std::cout << "]\n";
    } else {
      std::cout << "],\n";
    }
  }
  std::cout << "]\n";
}

int main(int argc, char *argv[]) {
  int ierr, totRank, mpiRank;

  // Initialize MPI
  ierr = MPI_Init(&argc, &argv);

  // Get the number of processes
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);

  // Get the local rank
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  if (!mpiRank) {
    cout << endl;
    cout << "-------------------" << endl;
    cout << " C2Decomp Testing " << endl;
    cout << "-------------------" << endl;
    cout << endl;
  }

  // We demonstrate the transposition with a pressure tensor. Our example domain
  // is a x b x c leading to a larger global size after
  // including the ghost points (for example a 2x2x2 global domain will result in a 3x5x5 domain. Refer to Constants for more infos).
  int nx = stoi(argv[2]), ny = stoi(argv[3]), nz = stoi(argv[4]);
  const int Pz = stoi(argv[1]);
  const int Py = totRank / Pz;
  if (!mpiRank)
    std::cout << "######################################## Pz: " << Pz
              << " Py: " << Py << "\n\n";
  assert(Pz * Py == totRank);
  assert(Pz > 0 && Py > 0);
  const mif::Constants constants(nx, ny, nz, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                                 1.0, 1, Py, Pz, mpiRank, {false, false, false});
  mif::StaggeredTensor local_tensor(constants, mif::StaggeringDirection::none);

  // Our local tensor will have these dimensions
  auto const n1 = constants.Nz;
  auto const n2 = constants.Ny;
  auto const n3 = constants.Nx;

  // Global sizes counting ghost cells
  int tot_Nx = n3, tot_Ny = 0, tot_Nz = 0;
  if (constants.z_rank == Pz - 1) {
    tot_Nz = ((n1 + 1) * (Pz - 1)) + (n1);
  } else {
    tot_Nz = (n1 * (Pz - 1)) + (n1 - 1);
  }
  if (constants.y_rank == Py - 1) {
    tot_Ny = ((n2 + 1) * (Py - 1)) + (n2);
  } else {
    tot_Ny = (n2 * (Py - 1)) + (n2 - 1);
  }
  if (!mpiRank) {
    std::cout
        << "######################################## common in ranks: tot Nx "
        << tot_Nx << " tot Ny " << tot_Ny << " tot Nz " << tot_Nz << "\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Initialize the tensor with easily recognizable values: set every Z face to
  // a incremental value
  int stride = constants.Nz;
  if (constants.z_rank == Pz - 1) {
    stride++;
  }
  for (size_t k = 0; k < constants.Nz; k++) {
    for (size_t j = 0; j < constants.Ny; j++) {
      for (size_t i = 0; i < constants.Nx; i++) {
        local_tensor(i, j, k) = (constants.z_rank * stride) + k;
      }
    }
  }

  // Show the tensor on all the processes
  MPI_Barrier(MPI_COMM_WORLD);
  for (int r = 0; r < totRank; ++r) {
    if (mpiRank == r) {
      std::cout << "rank " << mpiRank << " tensor (StaggeredTensor):\n";
      std::cout << local_tensor << "\n";
      print_t(local_tensor, n1, n2, n3);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Compute the global indices from the local indices. These will have to match
  // the X split of 2decomp
  const int start_x = 0;
  const int end_x = n3 - 1;
  const int start_z = constants.z_rank * (nz + 1);
  const int end_z = constants.z_rank * (nz + 1) + constants.Nz - 1;
  const int start_y = constants.y_rank * (ny + 1);
  const int end_y = constants.y_rank * (ny + 1) + constants.Ny - 1;

  if (!mpiRank) {
    cout << "\n######################################## Initializing 2decomp with sizes: " << tot_Nx << " x " << tot_Ny
         << " x " << tot_Nz << '\n';
  }
  C2Decomp *c2d;
  bool periodicBC[3] = {false, false, false};
  c2d = new C2Decomp(tot_Nx, tot_Ny, tot_Nz, Py, Pz, periodicBC);
  if (!mpiRank)
    cout << "######################################## Done initializing\n";

  MPI_Barrier(MPI_COMM_WORLD);

  // Create a global local_tensor for simpliciy. This have the same values as
  // the combined staggered tensors across the processors
  Real global_data[tot_Nz][tot_Ny][tot_Nx];
  for (int kp = 0; kp < tot_Nz; kp++) {
    for (int jp = 0; jp < tot_Ny; jp++) {
      for (int ip = 0; ip < tot_Nx; ip++) {
        global_data[kp][jp][ip] = static_cast<Real>(kp);
      }
    }
  }

  // Copy paste the output to: https://array-3d-viz.vercel.app/
  if (!mpiRank) {
    std::cout
        << "\nGlobal data (isualize in https://array-3d-viz.vercel.app/):\n";
    std::cout << "[\n";
    for (int kp = 0; kp < tot_Nz; kp++) {
      std::cout << "[";
      for (int jp = 0; jp < tot_Ny; jp++) {
        std::cout << "[";
        for (int ip = 0; ip < tot_Nx; ip++) {
          if (ip == tot_Nx - 1) {
            std::cout << global_data[kp][jp][ip];
          } else {
            std::cout << global_data[kp][jp][ip] << ',';
          }
        }
        if (jp == tot_Ny - 1) {
          std::cout << "]";

        } else {
          std::cout << "],";
        }
      }
      if (kp == tot_Nz - 1) {
        std::cout << "]\n";
      } else {
        std::cout << "],\n";
      }
    }
    std::cout << "]\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // These represent the local staggered tensor after being transposed in the ?
  // axe:
  // - xSize will contain the 3 axes sizes when the tensor is transposed in the
  // X direction
  // - ySize will contain the 3 axes sizes when the tensor is transposed in the
  // Y direction
  // - zSize will contain the 3 axes sizes when the tensor is transposed in the
  // Z direction
  //
  // - The global indices of start and end can be found in c2d->xStart, c2d->xEnd
  // for the X transposition
  // - The global indices of start and end can be found in
  // c2d->yStart, c2d->yEnd for the Y transposition
  // - The global indices of start
  // and end can be found in c2d->zStart, c2d->zEnd for the Z transposition
  Real xSize[3], ySize[3], zSize[3];
  xSize[0] = c2d->xSize[0];
  xSize[1] = c2d->xSize[1];
  xSize[2] = c2d->xSize[2];
  ySize[0] = c2d->ySize[0];
  ySize[1] = c2d->ySize[1];
  ySize[2] = c2d->ySize[2];
  zSize[0] = c2d->zSize[0];
  zSize[1] = c2d->zSize[1];
  zSize[2] = c2d->zSize[2];

  // Show splits
  for (int r = 0; r < totRank; ++r) {
    if (mpiRank == r) {
      std::cout << "\nrank: " << mpiRank << '\n';
      std::cout << "X size:\n";
      for (uint8_t i = 0; i < 3; ++i) {
        std::cout << xSize[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "X stard end:\n";
      for (uint8_t i = 0; i < 3; ++i) {
        std::cout << c2d->xStart[i] << '-' << c2d->xEnd[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Y size:\n";
      for (uint8_t i = 0; i < 3; ++i) {
        std::cout << ySize[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Y stard end:\n";
      for (uint8_t i = 0; i < 3; ++i) {
        std::cout << c2d->yStart[i] << '-' << c2d->yEnd[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Z size:\n";
      for (uint8_t i = 0; i < 3; ++i) {
        std::cout << zSize[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Z stard end:\n";
      for (uint8_t i = 0; i < 3; ++i) {
        std::cout << c2d->zStart[i] << '-' << c2d->zEnd[i] << ' ';
      }
      std::cout << "\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  Real *u1, *u2, *u3;
  Real t1, t2, t3;
  // auto const u1_size = c2d->allocX(u1);
  auto const u2_size = c2d->allocY(u2);
  auto const u3_size = c2d->allocZ(u3);

  // Copy from global data
  //for (int kp = 0; kp < xSize[2]; kp++) {
  //  for (int jp = 0; jp < xSize[1]; jp++) {
  //    for (int ip = 0; ip < xSize[0]; ip++) {
  //      int ii = kp * xSize[1] * xSize[0] + jp * xSize[0] + ip;
  //      // Copy data from local tensor
  //      u1[ii] = local_tensor(ip, jp, kp);
  //    }
  //  }
  //}
  //MPI_Barrier(MPI_COMM_WORLD);
  // This works but I don't know how safe it is...
  u1 = static_cast<Real*>(local_tensor.raw_data());

  // Show copied tensor on master rank
  if (mpiRank == 0) {
    std::cout << "rank " << mpiRank << " u1 tensor (2Decomp):\n";
    print_pointer(u1, xSize[2], xSize[1], xSize[0]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  Real transp_error = 0.0;

  t1 = MPI_Wtime();
  c2d->transposeX2Y_MajorIndex(u1, u2);
  t2 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("X2Y Elapsed time is %f\n", t2 - t1);
  }
  // Testing transposition
  MPI_Barrier(MPI_COMM_WORLD);
  for (int kp = 0; kp < ySize[2]; kp++) {
    for (int jp = 0; jp < ySize[1]; jp++) {
      for (int ip = 0; ip < ySize[0]; ip++) {
        int ii = ip * ySize[2] * ySize[1] + kp * ySize[1] + jp;
        Real temp = u2[ii];
        // Check with global data as after transposed, some data are in other
        // processors
        Real temp1 = global_data[c2d->yStart[2] + kp][c2d->yStart[1] + jp]
                                  [c2d->yStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  // Show transposed tensor on master rank
  if (mpiRank == 0) {
    std::cout << "rank " << mpiRank << " u2 tensor (2Decomp):\n";
    print_pointer(u2, ySize[2], ySize[1], ySize[0]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  t1 = MPI_Wtime();
  c2d->transposeY2Z_MajorIndex(u2, u3);
  t2 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("Y2Z Elapsed time is %f\n", t2 - t1);
  }
  // Testing transposition
  MPI_Barrier(MPI_COMM_WORLD);
  for (int kp = 0; kp < zSize[2]; kp++) {
    for (int jp = 0; jp < zSize[1]; jp++) {
      for (int ip = 0; ip < zSize[0]; ip++) {
        int ii = jp * zSize[2] * zSize[0] + ip * zSize[2] + kp;
        Real temp = u3[ii];
        Real temp1 = global_data[c2d->zStart[2] + kp][c2d->zStart[1] + jp]
                                  [c2d->zStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  // Show transposed tensor on master rank
  if (mpiRank == 0) {
    std::cout << "rank " << mpiRank << " u3 tensor (2Decomp):\n";
    print_pointer(u3, zSize[2], zSize[1], zSize[0]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  t1 = MPI_Wtime();
  c2d->transposeZ2Y_MajorIndex(u3, u2);
  t2 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("Z2Y Elapsed time is %f\n", t2 - t1);
  }
  // Testing transposition
  MPI_Barrier(MPI_COMM_WORLD);
  for (int kp = 0; kp < ySize[2]; kp++) {
    for (int jp = 0; jp < ySize[1]; jp++) {
      for (int ip = 0; ip < ySize[0]; ip++) {
        int ii = ip * ySize[2] * ySize[1] + kp * ySize[1] + jp;
        Real temp = u2[ii];
        Real temp1 = global_data[c2d->yStart[2] + kp][c2d->yStart[1] + jp]
                                  [c2d->yStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  // Show transposed tensor on master rank
  if (mpiRank == 0) {
    std::cout << "rank " << mpiRank << " u2 tensor (2Decomp):\n";
    print_pointer(u2, ySize[2], ySize[1], ySize[0]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  t1 = MPI_Wtime();
  c2d->transposeY2X_MajorIndex(u2, u1);
  t2 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("Y2X Elapsed time is %f\n", t2 - t1);
  }
  // Testing transposition
  MPI_Barrier(MPI_COMM_WORLD);
  for (int kp = 0; kp < xSize[2]; kp++) {
    for (int jp = 0; jp < xSize[1]; jp++) {
      for (int ip = 0; ip < xSize[0]; ip++) {
        int ii = kp * xSize[1] * xSize[0] + jp * xSize[0] + ip;
        Real temp = u1[ii];
        Real temp1 = global_data[c2d->xStart[2] + kp][c2d->xStart[1] + jp]
                                  [c2d->xStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  // Show transposed tensor on master rank
  if (mpiRank == 0) {
    std::cout << "rank " << mpiRank << " u1 tensor (2Decomp):\n";
    print_pointer(u1, xSize[2], xSize[1], xSize[0]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // allocate new buffers for non-blocking comms
  Real *sbuf = new Real[c2d->decompBufSize];
  Real *rbuf = new Real[c2d->decompBufSize];
  MPI_Request x2yHandle;
  t1 = MPI_Wtime();
  c2d->transposeX2Y_MajorIndex_Start(x2yHandle, u1, u2, sbuf, rbuf);
  t2 = MPI_Wtime();
  c2d->transposeX2Y_MajorIndex_Wait(x2yHandle, u1, u2, sbuf, rbuf);
  t3 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("X2Y Nonblocking Start Elapsed time is %f\n", t2 - t1);
    printf("X2Y Nonblocking Wait Elapsed time is %f\n", t3 - t2);
  }
  // Testing transposition
  for (int kp = 0; kp < ySize[2]; kp++) {
    for (int jp = 0; jp < ySize[1]; jp++) {
      for (int ip = 0; ip < ySize[0]; ip++) {
        int ii = ip * ySize[1] * ySize[2] + kp * ySize[1] + jp;
        Real temp = u2[ii];
        Real temp1 = global_data[c2d->yStart[2] + kp][c2d->yStart[1] + jp]
                                  [c2d->yStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  MPI_Request y2zHandle;
  t1 = MPI_Wtime();
  c2d->transposeY2Z_MajorIndex_Start(y2zHandle, u2, u3, sbuf, rbuf);
  t2 = MPI_Wtime();
  c2d->transposeY2Z_MajorIndex_Wait(y2zHandle, u2, u3, sbuf, rbuf);
  t3 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("Y2Z Nonblocking Start Elapsed time is %f\n", t2 - t1);
    printf("Y2Z Nonblocking Wait Elapsed time is %f\n", t3 - t2);
  }
  // Testing transposition
  for (int kp = 0; kp < zSize[2]; kp++) {
    for (int jp = 0; jp < zSize[1]; jp++) {
      for (int ip = 0; ip < zSize[0]; ip++) {
        int ii = jp * zSize[2] * zSize[0] + ip * zSize[2] + kp;
        Real temp = u3[ii];
        Real temp1 = global_data[c2d->zStart[2] + kp][c2d->zStart[1] + jp]
                                  [c2d->zStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  MPI_Request z2yHandle;
  t1 = MPI_Wtime();
  c2d->transposeZ2Y_MajorIndex_Start(z2yHandle, u3, u2, sbuf, rbuf);
  t2 = MPI_Wtime();
  c2d->transposeZ2Y_MajorIndex_Wait(z2yHandle, u3, u2, sbuf, rbuf);
  t3 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("Z2Y Nonblocking Start Elapsed time is %f\n", t2 - t1);
    printf("Z2Y Nonblocking Wait Elapsed time is %f\n", t3 - t2);
  }
  // Testing transposition
  for (int kp = 0; kp < ySize[2]; kp++) {
    for (int jp = 0; jp < ySize[1]; jp++) {
      for (int ip = 0; ip < ySize[0]; ip++) {
        int ii = ip * ySize[1] * ySize[2] + kp * ySize[1] + jp;
        Real temp = u2[ii];
        Real temp1 = global_data[c2d->yStart[2] + kp][c2d->yStart[1] + jp]
                                  [c2d->yStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  MPI_Request y2xHandle;
  t1 = MPI_Wtime();
  c2d->transposeY2X_MajorIndex_Start(y2xHandle, u2, u1, sbuf, rbuf);
  t2 = MPI_Wtime();
  c2d->transposeY2X_MajorIndex_Wait(y2xHandle, u2, u1, sbuf, rbuf);
  t3 = MPI_Wtime();
  if (mpiRank == 0) {
    printf("Y2X Nonblocking Start Elapsed time is %f\n", t2 - t1);
    printf("Y2X Nonblocking Wait Elapsed time is %f\n", t3 - t2);
  }

  // Testing transposition
  for (int kp = 0; kp < xSize[2]; kp++) {
    for (int jp = 0; jp < xSize[1]; jp++) {
      for (int ip = 0; ip < xSize[0]; ip++) {
        int ii = kp * xSize[1] * xSize[0] + jp * xSize[0] + ip;
        Real temp = u1[ii];
        Real temp1 = global_data[c2d->xStart[2] + kp][c2d->xStart[1] + jp]
                                  [c2d->xStart[0] + ip];
        transp_error += abs(temp - temp1);
      }
    }
  }

  delete[] sbuf;
  delete[] rbuf;
  // c2d->deallocXYZ(u1);
  c2d->deallocXYZ(u2);
  c2d->deallocXYZ(u3);

  delete (c2d);

  for (unsigned i = 0; i < totRank; ++i) {
    if (mpiRank == i) {
      std::cout << "rank " << i << " transposition error: " << transp_error
                << '\n';
      assert(transp_error < 1E-6);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Now lets kill MPI
  MPI_Finalize();

  return 0;
}
