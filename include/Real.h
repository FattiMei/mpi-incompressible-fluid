#ifndef REAL_H
#define REAL_H

// A way to make the code independent of the data type used for floating point
// numbers.

namespace mif {

#define Real double
#define MPI_MIF_REAL MPI_DOUBLE
#if  not USE_DOUBLE
#undef Real
#define Real float
#undef MPI_MIF_REAL
#define MPI_MIF_REAL MPI_FLOAT

// #undef MPI_Real_PRECISION
// #define MPI_Real_PRECISION MPI_FLOAT_PRECISION
#endif
} // namespace mif

#endif // REAL_H