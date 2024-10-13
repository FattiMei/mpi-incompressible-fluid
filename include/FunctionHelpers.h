#ifndef FUNCTION_HELPERS_H
#define FUNCTION_HELPERS_H

#include <functional>
#include "Constants.h"
#include "Real.h"
#include "Tensor.h"
#include "VelocityComponent.h"

namespace mif {

    // Return an input function f, depending on x,y,z and time, removing its time dependency.
    inline std::function<Real(Real, Real, Real)> function_at_time(const std::function<Real(Real, Real, Real, Real)> &f, Real time) {
        const std::function<Real(Real, Real, Real)> result = 
                [&time, &f](Real x, Real y, Real z) {
                    return f(time, x, y, z);
                };
        return result;
    }

    // Return a reference to the input tensor corresponding to the given component of the velocity.
    template <VelocityComponent component> inline const Tensor<>& 
    choose_component(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w) {
        if constexpr (component == VelocityComponent::u) {
            return u;
        } else if constexpr (component == VelocityComponent::v) {
            return v;
        } else {
            static_assert(component == VelocityComponent::w);
            return w;
        }
    }

    // Evaluate a function on a staggered mesh corresponding to a component of velocity.
    template <VelocityComponent component> inline Real
    evaluate_staggered(const Tensor<> &tensor, size_t i, size_t j, size_t k, 
                       const std::function<Real(Real, Real, Real)> &f, 
                       const Constants &constants) {
        if constexpr (component == VelocityComponent::u) {
            return f(constants.dx * i + constants.dx * 0.5, constants.dy * j, constants.dz * k);
        } else if constexpr (component == VelocityComponent::v) {
            return f(constants.dx * i, constants.dy * j + constants.dy * 0.5, constants.dz * k);
        } else {
            static_assert(component == VelocityComponent::w);
            return f(constants.dx * i, constants.dy * j, constants.dz * k + constants.dz * 0.5);
        }
    }

} // mif

#endif // FUNCTION_HELPERS_H