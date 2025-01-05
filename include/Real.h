#ifndef REAL_H
#define REAL_H

// A way to make the code independent of the data type used for floating point
// numbers.

namespace mif {

#if not USE_DOUBLE
#undef Real
#define Real float
#undef MPI_MIF_REAL
#define MPI_MIF_REAL MPI_FLOAT
#else
#define Real double
#define MPI_MIF_REAL MPI_DOUBLE
#endif

} // namespace mif

#endif // REAL_H