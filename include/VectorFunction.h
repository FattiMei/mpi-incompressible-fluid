#ifndef VECTOR_FUNCTION_H
#define VECTOR_FUNCTION_H

#include <functional>
#include "Real.h"

namespace mif {

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