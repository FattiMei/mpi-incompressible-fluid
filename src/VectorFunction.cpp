#include "FunctionHelpers.h"
#include "VectorFunction.h"

namespace mif {

    VectorFunction::VectorFunction(const std::function<Real(Real, Real, Real)> f_u,
                                   const std::function<Real(Real, Real, Real)> f_v,
                                   const std::function<Real(Real, Real, Real)> f_w):
            f_u(f_u), f_v(f_v), f_w(f_w),
            components{&this->f_u, &this->f_v, &this->f_w} {};

    TimeVectorFunction::TimeVectorFunction(const std::function<Real(Real, Real, Real, Real)> f_u,
                                           const std::function<Real(Real, Real, Real, Real)> f_v,
                                           const std::function<Real(Real, Real, Real, Real)> f_w): 
            f_u(f_u), f_v(f_v), f_w(f_w),
            components{&this->f_u, &this->f_v, &this->f_w} {};

    VectorFunction TimeVectorFunction::set_time(Real time) const {
        return VectorFunction(function_at_time(f_u, time),
                              function_at_time(f_v, time),
                              function_at_time(f_w, time));
    }

} // mif