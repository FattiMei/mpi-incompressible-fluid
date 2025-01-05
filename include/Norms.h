#ifndef NORMS_H
#define NORMS_H

#include "VectorFunction.h"
#include "VelocityTensor.h"

namespace mif {

// All norms use second order schemes and assume no error on the boundaries
// (or equivalently Dirichlet boundary conditions on all boundaries).
// The error is calculated using the functions in Manufactured.h.
// In parallel, this still makes sense, since real boundaries still have no 
// error, and processor boundaries already have their contribution in 
// neighbouring processors' errors. Errors on periodic boundaries are included
// in the error computation.

// The L1 norm of a vector function is defined as the integral over the whole
// domain of the norm of the vector.
Real ErrorL1Norm(const VelocityTensor &velocity, const TimeVectorFunction &exact_velocity, Real time);

// The L2 norm of a vector function is defined as the square root of the
// integral over the whole domain of the scalar product of the function with
// itself.
Real ErrorL2Norm(const VelocityTensor &velocity, const TimeVectorFunction &exact_velocity, Real time);

// The Linfinity norm of a vector function is defined as the maximum value of
// any component of the function.
Real ErrorLInfNorm(const VelocityTensor &velocity, const TimeVectorFunction &exact_velocity, Real time);

// Versions of the same functions for a scalar tensor.
Real ErrorL1Norm(const StaggeredTensor &pressure, const std::function<Real(Real, Real, Real, Real)> &exact_pressure, Real time);
Real ErrorL2Norm(const StaggeredTensor &pressure, const std::function<Real(Real, Real, Real, Real)> &exact_pressure, Real time);
Real ErrorLInfNorm(const StaggeredTensor &pressure, const std::function<Real(Real, Real, Real, Real)> &exact_pressure, Real time);

// Compute the global error in a given norm by accumulating the errors on
// the processor with rank 0. For all other processors the result will be -1.
// If not all processors call this function, it may result in an error or
// deadlock. This function works for the L1 norm.
Real accumulate_error_mpi_l1(Real local_error, const Constants &constants);

// Same as the previous function, but for the L2 norm.
Real accumulate_error_mpi_l2(Real local_error, const Constants &constants);

// Same as the previous function, but for the Linf norm.
Real accumulate_error_mpi_linf(Real local_error, const Constants &constants);

} // namespace mif

#endif // NORMS_H