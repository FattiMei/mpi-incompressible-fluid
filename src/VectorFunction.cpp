#include "VectorFunction.h"

namespace mif {

// Given a fixed time, this utility removes the time dependency of a
// function f(x,y,z,t) by substituting the time argument with the fixed
// time. This is used in the set_time method of TimeVectorFunction.
inline std::function<Real(Real, Real, Real)>
function_at_time(const std::function<Real(Real, Real, Real, Real)> &f,
                 Real time) {
  return [time, &f](Real x, Real y, Real z) { return f(time, x, y, z); };
}

VectorFunction::VectorFunction(const std::function<Real(Real, Real, Real)> f_u,
                               const std::function<Real(Real, Real, Real)> f_v,
                               const std::function<Real(Real, Real, Real)> f_w)
    : f_u(f_u), f_v(f_v), f_w(f_w),
      components{&this->f_u, &this->f_v, &this->f_w} {};

TimeVectorFunction::TimeVectorFunction(
    const std::function<Real(Real, Real, Real, Real)> f_u,
    const std::function<Real(Real, Real, Real, Real)> f_v,
    const std::function<Real(Real, Real, Real, Real)> f_w)
    : f_u(f_u), f_v(f_v), f_w(f_w),
      components{&this->f_u, &this->f_v, &this->f_w} {};

VectorFunction TimeVectorFunction::set_time(Real time) const {
  return VectorFunction(function_at_time(f_u, time),
                        function_at_time(f_v, time),
                        function_at_time(f_w, time));
}

VectorFunction TimeVectorFunction::get_difference_over_time(Real time_1, Real time_2) const {
  const std::function<Real(Real, Real, Real)> new_f_u = 
      [time_1, time_2, this](Real x, Real y, Real z) { return f_u(time_2, x, y, z) - f_u(time_1, x, y, z); };
  const std::function<Real(Real, Real, Real)> new_f_v = 
      [time_1, time_2, this](Real x, Real y, Real z) { return f_v(time_2, x, y, z) - f_v(time_1, x, y, z); };
  const std::function<Real(Real, Real, Real)> new_f_w = 
      [time_1, time_2, this](Real x, Real y, Real z) { return f_w(time_2, x, y, z) - f_w(time_1, x, y, z); };
  return VectorFunction(new_f_u, new_f_v, new_f_w);
}

} // namespace mif