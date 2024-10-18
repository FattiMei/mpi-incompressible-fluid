#ifndef VECTOR_FUNCTION_H
#define VECTOR_FUNCTION_H

#include <functional>
#include "Real.h"

namespace mif {

    // A collection of 3 functions with 3 Real inputs (x,y,z) and one Real output.
    class VectorFunction {
      public:

        VectorFunction(const std::function<Real(Real, Real, Real)> f_u,
                       const std::function<Real(Real, Real, Real)> f_v,
                       const std::function<Real(Real, Real, Real)> f_w);

        const std::function<Real(Real, Real, Real)> f_u;
        const std::function<Real(Real, Real, Real)> f_v;
        const std::function<Real(Real, Real, Real)> f_w;
        const std::array<const std::function<Real(Real, Real, Real)>*, 3> components;
    };

    // A collection of 3 functions with 4 Real inputs (t,x,y,z) and one Real output.
    class TimeVectorFunction {
      public:

        TimeVectorFunction(const std::function<Real(Real, Real, Real, Real)> f_u,
                           const std::function<Real(Real, Real, Real, Real)> f_v,
                           const std::function<Real(Real, Real, Real, Real)> f_w);

        const std::function<Real(Real, Real, Real, Real)> f_u;
        const std::function<Real(Real, Real, Real, Real)> f_v;
        const std::function<Real(Real, Real, Real, Real)> f_w;
        const std::array<const std::function<Real(Real, Real, Real, Real)>*, 3> components;

        VectorFunction set_time(Real time) const;
    };

} // mif

#endif // VECTOR_FUNCTION_H