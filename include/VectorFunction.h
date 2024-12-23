#ifndef VECTOR_FUNCTION_H
#define VECTOR_FUNCTION_H

#include "Real.h"
#include <functional>

// Since we often need to carry around a collection of functions, namely the
// three components of a vector field at a point in space, we define classes
// VectorFunction and TimeVectorFunction to hold such collections.

namespace mif {

// A collection of 3 functions with 3 Real inputs (x,y,z) and one Real
// output. This can be seen as a single function from R^3 to R^3.
class VectorFunction {
public:
  VectorFunction(const std::function<Real(Real, Real, Real)> f_u,
                 const std::function<Real(Real, Real, Real)> f_v,
                 const std::function<Real(Real, Real, Real)> f_w);

  const std::function<Real(Real, Real, Real)> f_u;
  const std::function<Real(Real, Real, Real)> f_v;
  const std::function<Real(Real, Real, Real)> f_w;
  const std::array<const std::function<Real(Real, Real, Real)> *, 3> components;
};

// A collection of 3 functions with 4 Real inputs (t,x,y,z) and one Real
// output. This can be seen as a single function from R^4 to R^3, and is
// used to represent a time-dependent vector field.
class TimeVectorFunction {
public:
  TimeVectorFunction(const std::function<Real(Real, Real, Real, Real)> f_u,
                     const std::function<Real(Real, Real, Real, Real)> f_v,
                     const std::function<Real(Real, Real, Real, Real)> f_w);

  const std::function<Real(Real, Real, Real, Real)> f_u;
  const std::function<Real(Real, Real, Real, Real)> f_v;
  const std::function<Real(Real, Real, Real, Real)> f_w;
  const std::array<const std::function<Real(Real, Real, Real, Real)> *, 3>
      components;

  // This method projects the TimeVectorFunction to a VectorFunction at a
  // given time, by removing the time dependency.
  VectorFunction set_time(Real time) const;

  // Return a VectorFunction whose entries are the entries of this function
  // after setting time_2, minus the entries of this function after setting
  // time_1.
  VectorFunction get_difference_over_time(Real time_1, Real time_2) const;
};

} // namespace mif

#endif // VECTOR_FUNCTION_H