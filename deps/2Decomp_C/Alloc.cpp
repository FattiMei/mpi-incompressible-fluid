#include "C2Decomp.hpp"

int C2Decomp::allocX(Real *&var) {

  int xsize = decompMain.xsz[0];
  int ysize = decompMain.xsz[1];
  int zsize = decompMain.xsz[2];

  var = new Real[xsize * ysize * zsize];
  return xsize * ysize * zsize;
}

int C2Decomp::allocY(Real *&var) {

  int xsize = decompMain.ysz[0];
  int ysize = decompMain.ysz[1];
  int zsize = decompMain.ysz[2];

  var = new Real[xsize * ysize * zsize];
  return xsize * ysize * zsize;
}

int C2Decomp::allocZ(Real *&var) {

  int xsize = decompMain.zsz[0];
  int ysize = decompMain.zsz[1];
  int zsize = decompMain.zsz[2];

  var = new Real[xsize * ysize * zsize];
  return xsize * ysize * zsize;
}

void C2Decomp::deallocXYZ(Real *&var) {
  delete[] var;
  var = NULL;
}
