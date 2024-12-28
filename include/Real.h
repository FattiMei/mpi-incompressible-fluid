#ifndef REAL_H
#define REAL_H

// A way to make the code independent of the data type used for floating point
// numbers.

namespace mif {

#define Real double
#define MPI_MIF_REAL MPI_DOUBLE

} // namespace mif

#endif // REAL_H