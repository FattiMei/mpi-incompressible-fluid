#include "VectorFunction.h"

namespace mif {
    // Given a fixed time, this utility removes the time dependency of a
    // function f(x,y,z,t) by substituting the time argument with the fixed
    // time. This is used in the set_time method of TimeVectorFunction.



    VectorFunction::VectorFunction(const Real (*f_u)(Real, Real, Real) noexcept,
                                   const Real (*f_v)(Real, Real, Real) noexcept,
                                   const Real (*f_w)(Real, Real, Real)noexcept)
        : f_u(f_u), f_v(f_v), f_w(f_w),
          components{&this->f_u, &this->f_v, &this->f_w} {
    };

    TimeVectorFunction::TimeVectorFunction(
        const Real (*f_u)(Real, Real, Real,Real)noexcept,
        const Real (*f_v)(Real, Real, Real,Real)noexcept,
        const Real (*f_w)(Real, Real, Real,Real)noexcept)
        : f_u(f_u), f_v(f_v), f_w(f_w),
          components{&this->f_u, &this->f_v, &this->f_w} {
    };


    /*VectorFunction TimeVectorFunction::set_time(Real time) const {
        return VectorFunction(function_at_time(f_u, time),
                              function_at_time(f_v, time),
                              function_at_time(f_w, time));

    }*/
} // namespace mif
