#ifndef NORMS_H
#define NORMS_H

#include "VelocityTensor.h"

namespace mif {

// All norms use second order schemes and assume no error on the boundaries
// (or equivalently Dirichlet boundary conditions on all boundaries).
// The error is calculated using the functions in Manufactured.h.

// The L1 norm of a vector function is defined as the integral over the whole
// domain of the norm of the vector.
Real ErrorL1Norm(const VelocityTensor &velocity, Real time);

// The L2 norm of a vector function is defined as the square root of the
// integral over the whole domain of the scalar product of the function with
// itself.
Real ErrorL2Norm(const VelocityTensor &velocity, Real time);

// The Linfinity norm of a vector function is defined as the maximum value of
// any component of the function.
Real ErrorLInfNorm(const VelocityTensor &velocity, Real time);

} // namespace mif

#endif // NORMS_H