#ifndef FUNCTION_HELPERS_H
#define FUNCTION_HELPERS_H

#include <functional>
#include "Real.h"

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

} // mif

#endif // FUNCTION_HELPERS_H