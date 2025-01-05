#ifndef _2DECOMPCH_
#define _2DECOMPCH_

#include "math.h"
#include "mpi.h"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory.h>
#include <string>
#include "Real.h"

using namespace ::std;

class C2Decomp {

public:
  enum Distribution { DEFAULT = 0, MIF = 1 };

  // Just assume that we're using Real precision all of the time
  typedef Real myType;
  MPI_Datatype realType;
  int myTypeBytes;

  // Global Size
  int nxGlobal, nyGlobal, nzGlobal;

  // MPI rank info
  int nRank, nProc;

public:
  // parameters for 2D Cartesian Topology
  int dims[2], coord[2];
  int periodic[2];

public:
  MPI_Comm DECOMP_2D_COMM_CART_X, DECOMP_2D_COMM_CART_Y, DECOMP_2D_COMM_CART_Z;
  MPI_Comm DECOMP_2D_COMM_ROW, DECOMP_2D_COMM_COL;

private:
  // Defining neighboring blocks
  int neighbor[3][6];
  // Flags for periodic condition in 3D
  bool periodicX, periodicY, periodicZ;
  // Distribution type
  enum Distribution distributionType;

public:
  // Struct used to store decomposition info for a given global data size
  typedef struct decompinfo {
    int xst[3], xen[3], xsz[3];
    int yst[3], yen[3], ysz[3];
    int zst[3], zen[3], zsz[3];

    int *x1dist, *y1dist, *y2dist, *z2dist;
    int *x1cnts, *y1cnts, *y2cnts, *z2cnts;
    int *x1disp, *y1disp, *y2disp, *z2disp;

    int x1count, y1count, y2count, z2count;

    bool even;
  } DecompInfo;

public:
  // main default decomposition information for global size nx*ny*nz
  DecompInfo decompMain;
  int decompBufSize;

public:
  // Starting/ending index and size of data held by the current processor
  // duplicate 'decompMain', needed by apps to define data structure
  int xStart[3], xEnd[3], xSize[3]; // x-pencil
  int yStart[3], yEnd[3], ySize[3]; // y-pencil
  int zStart[3], zEnd[3], zSize[3]; // z-pencil

private:
  // These are the buffers used by MPI_ALLTOALL(V) calls
  Real *work1_r, *work2_r; // Only implementing real for now...

public:
  void decomp2DInit(int pRow, int pCol);

  template <Distribution disType = Distribution::MIF>
  C2Decomp(int nx, int ny, int nz, int pRow, int pCol, bool periodicBC[3]) {

    nxGlobal = nx;
    nyGlobal = ny;
    nzGlobal = nz;

    periodicX = periodicBC[0];
    periodicY = periodicBC[1];
    periodicZ = periodicBC[2];

    decompBufSize = 0;
    work1_r = NULL;
    work2_r = NULL;

    realType = MPI_MIF_REAL;

    distributionType = disType;

    decomp2DInit(pRow, pCol);
  }

  void best2DGrid(int nProc, int &pRow, int &pCol);
  void FindFactor(int num, int *factors, int &nfact);

  void decomp2DFinalize();

  // Just get it running without the optional decomp for now...
  void transposeX2Y(Real* src, Real* dst);
  void transposeY2Z(Real* src, Real* dst);
  void transposeZ2Y(Real* src, Real* dst);
  void transposeY2X(Real* src, Real* dst);

  // Get Transposes but with array indexing with the major index of the
  // pencil...
  void transposeX2Y_MajorIndex(Real* src, Real* dst);
  void transposeY2Z_MajorIndex(Real* src, Real* dst);
  void transposeZ2Y_MajorIndex(Real* src, Real* dst);
  void transposeY2X_MajorIndex(Real* src, Real* dst);

  // calls for overlapping communication and computation...
  void transposeX2Y_Start(MPI_Request& handle, Real* src, Real* dst,
                          Real* sbuf, Real* rbuf);
  void transposeX2Y_Wait(MPI_Request& handle, Real* src, Real* dst,
                         Real* sbuf, Real* rbuf);

  void transposeY2Z_Start(MPI_Request& handle, Real* src, Real* dst,
                          Real* sbuf, Real* rbuf);
  void transposeY2Z_Wait(MPI_Request& handle, Real* src, Real* dst,
                         Real* sbuf, Real* rbuf);

  void transposeZ2Y_Start(MPI_Request& handle, Real* src, Real* dst,
                          Real* sbuf, Real* rbuf);
  void transposeZ2Y_Wait(MPI_Request& handle, Real* src, Real* dst,
                         Real* sbuf, Real* rbuf);

  void transposeY2X_Start(MPI_Request& handle, Real* src, Real* dst,
                          Real* sbuf, Real* rbuf);
  void transposeY2X_Wait(MPI_Request& handle, Real* src, Real* dst,
                         Real* sbuf, Real* rbuf);

  // calls for overlapping communication and computation...
  void transposeX2Y_MajorIndex_Start(MPI_Request& handle, Real* src,
                                     Real* dst, Real* sbuf, Real* rbuf);
  void transposeX2Y_MajorIndex_Wait(MPI_Request& handle, Real* src,
                                    Real* dst, Real* sbuf, Real* rbuf);

  void transposeY2Z_MajorIndex_Start(MPI_Request& handle, Real* src,
                                     Real* dst, Real* sbuf, Real* rbuf);
  void transposeY2Z_MajorIndex_Wait(MPI_Request& handle, Real* src,
                                    Real* dst, Real* sbuf, Real* rbuf);

  void transposeZ2Y_MajorIndex_Start(MPI_Request& handle, Real* src,
                                     Real* dst, Real* sbuf, Real* rbuf);
  void transposeZ2Y_MajorIndex_Wait(MPI_Request& handle, Real* src,
                                    Real* dst, Real* sbuf, Real* rbuf);

  void transposeY2X_MajorIndex_Start(MPI_Request& handle, Real* src,
                                     Real* dst, Real* sbuf, Real* rbuf);
  void transposeY2X_MajorIndex_Wait(MPI_Request& handle, Real* src,
                                    Real* dst, Real* sbuf, Real* rbuf);

  void decompInfoInit();
  void decompInfoFinalize();

  // only doing real
  int allocX(Real*& var);
  int allocY(Real*& var);
  int allocZ(Real*& var);
  void deallocXYZ(Real*& var);

  void updateHalo(Real* in, Real*& out, int level, int ipencil);

  void decomp2DAbort(int errorCode, string msg);
  void initNeighbor();
  void getDist();
  void distribute(int data1, int proc, int *st, int *en, int *sz);
  void partition(int nx, int ny, int nz, int *pdim, int *lstart, int *lend,
                 int *lsize);
  void prepareBuffer(DecompInfo *dii);

  void getDecompInfo(DecompInfo dcompinfo_in);

  void memSplitXY(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);

  void memMergeXY(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);
  void memMergeXY_YMajor(Real* in, int n1, int n2, int n3, Real* out,
                         int iproc, int *dist);

  void memSplitYZ(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);
  void memSplitYZ_YMajor(Real* in, int n1, int n2, int n3, Real* out,
                         int iproc, int *dist);

  void memMergeYZ(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);
  void memMergeYZ_ZMajor(Real* in, int n1, int n2, int n3, Real* out,
                         int iproc, int *dist);

  void memSplitZY(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);
  void memSplitZY_ZMajor(Real* in, int n1, int n2, int n3, Real* out,
                         int iproc, int *dist);

  void memMergeZY(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);
  void memMergeZY_YMajor(Real* in, int n1, int n2, int n3, Real* out,
                         int iproc, int *dist);

  void memSplitYX(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);
  void memSplitYX_YMajor(Real* in, int n1, int n2, int n3, Real* out,
                         int iproc, int *dist);

  void memMergeYX(Real* in, int n1, int n2, int n3, Real* out, int iproc,
                  int *dist);

  // IO
  void writeOne(int ipencil, Real* var, string filename);
  void writeVar(MPI_File& fh, MPI_Offset& disp, int ipencil, Real* var);
  void writeScalar(MPI_File& fh, MPI_Offset& disp, int n, Real* var);
  void writePlane(int ipencil, Real* var, int iplane, int n, string filename);
  void writeEvery(int ipencil, Real* var, int iskip, int jskip, int kskip,
                  string filename, bool from1);

  void readOne(int ipencil, Real* var, string filename);
  void readVar(MPI_File& fh, MPI_Offset& disp, int ipencil, Real* var);
  void readScalar(MPI_File& fh, MPI_Offset& disp, int n, Real* var);
};

#endif
