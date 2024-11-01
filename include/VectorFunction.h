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
  VectorFunction(const  Real (*f_u)(Real, Real, Real)noexcept,
                 const  Real (*f_v)(Real, Real, Real)noexcept,
                 const  Real (*f_w)(Real, Real, Real)noexcept);

  const  Real (*f_u)(Real, Real, Real)noexcept;
  const Real (*f_v)(Real, Real, Real)noexcept;
  const  Real (*f_w)(Real, Real, Real)noexcept;
  const std::array<const Real (**)(Real, Real, Real) noexcept, 3> components;
};

// A collection of 3 functions with 4 Real inputs (t,x,y,z) and one Real
// output. This can be seen as a single function from R^4 to R^3, and is
// used to represent a time-dependent vector field.
class TimeVectorFunction {
public:
    Real time;
    TimeVectorFunction(const  Real (*f_u)(Real, Real, Real,Real)noexcept,
                   const  Real (*f_v)(Real, Real, Real,Real)noexcept,
                   const  Real (*f_w)(Real, Real, Real,Real)noexcept);

   const  Real (*f_u)(Real, Real, Real,Real)noexcept;
    const  Real (*f_v)(Real, Real, Real,Real)noexcept;
    const  Real (*f_w)(Real, Real, Real,Real)noexcept;
  const std::array<const Real (**)(Real, Real, Real,Real) noexcept, 3>
      components;

  // This method projects the TimeVectorFunction to a VectorFunction at a
  // given time, by removing the time dependency.
  // VectorFunction set_time(Real time) const;

};

} // namespace mif

#endif // VECTOR_FUNCTION_H