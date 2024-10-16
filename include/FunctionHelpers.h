#ifndef FUNCTION_HELPERS_H
#define FUNCTION_HELPERS_H

#include <functional>
#include "Constants.h"
#include "Real.h"
#include "Tensor.h"
#include "VelocityComponent.h"

namespace mif {

    // Return an input function f, depending on x,y,z and time, removing its time dependency.
    inline std::function<Real(Real, Real, Real)> 
    function_at_time(const std::function<Real(Real, Real, Real, Real)> &f, Real time) {
        const std::function<Real(Real, Real, Real)> result = 
                [time, &f](Real x, Real y, Real z) {
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

    // Return a lambda function corresponding to the given component of the velocity.
    template <VelocityComponent component, typename MaybeCallable> inline const MaybeCallable& 
    choose_function(const MaybeCallable &f_u, const MaybeCallable &f_v, const MaybeCallable &f_w) {
        if constexpr (component == VelocityComponent::u) {
            return f_u;
        } else if constexpr (component == VelocityComponent::v) {
            return f_v;
        } else {
            static_assert(component == VelocityComponent::w);
            return f_w;
        }
    }

    // Evaluate a function on a staggered mesh corresponding to a component of velocity.
    template <VelocityComponent component> inline Real
    evaluate_staggered(size_t i, size_t j, size_t k, 
                       const std::function<Real(Real, Real, Real)> &f, 
                       const Constants &constants) {
        assert(i >= 0 && i < constants.Nx);
        assert(j >= 0 && j < constants.Ny);
        assert(k >= 0 && k < constants.Nz);
        if constexpr (component == VelocityComponent::u) {
            assert(i < constants.Nx-1);
            return f(constants.dx * i + constants.dx * 0.5, constants.dy * j, constants.dz * k);
        } else if constexpr (component == VelocityComponent::v) {
            assert(j < constants.Ny-1);
            return f(constants.dx * i, constants.dy * j + constants.dy * 0.5, constants.dz * k);
        } else {
            assert(k < constants.Nz-1);
            static_assert(component == VelocityComponent::w);
            return f(constants.dx * i, constants.dy * j, constants.dz * k + constants.dz * 0.5);
        }
    }

    // Apply function f to all points of a tensor representing a velocity component.
    // The function must take as parameters the tensor, the indices i,j,k, constants and possibly the arguments args.
    template <VelocityComponent component, typename MaybeCallable, typename... MaybeCallableArgs> inline void
    apply_on_all_points(Tensor<> &tensor, const MaybeCallable &f, MaybeCallableArgs... args) {
        for (size_t i = 0; i < tensor.sizes()[0]; i++) {
            for (size_t j = 0; j < tensor.sizes()[1]; j++) {
                for (size_t k = 0; k < tensor.sizes()[2]; k++) {
                    f(tensor, i, j, k, args...);
                }
            }
        }
    }

    // Apply function f to all internal points of a tensor representing a velocity component.
    // The function must take as parameters the tensor, the indices i,j,k, constants and possibly the arguments args.
    template <VelocityComponent component, typename MaybeCallable, typename... MaybeCallableArgs> inline void
    apply_on_internal_points(Tensor<> &tensor, const MaybeCallable &f, MaybeCallableArgs... args) {
        for (size_t i = 1; i < tensor.sizes()[0]-1; i++) {
            for (size_t j = 1; j < tensor.sizes()[1]-1; j++) {
                for (size_t k = 1; k < tensor.sizes()[2]-1; k++) {
                    f(tensor, i, j, k, args...);
                }
            }
        }
    }

    // Compute the function f on all points of a tensor representing a velocity component, and store the results in the tensor.
    template <VelocityComponent component> void 
    discretize_function(Tensor<> &tensor, const std::function<Real(Real, Real, Real)> &f, const Constants &constants) {
        const auto applicable_f = [&f, &constants](Tensor<>& tensor, size_t i, size_t j, size_t k) {
            tensor(i,j,k) = evaluate_staggered<component>(i, j, k, f, constants);
        };
        apply_on_all_points<component>(tensor, applicable_f);
    }

} // mif

#endif // FUNCTION_HELPERS_H