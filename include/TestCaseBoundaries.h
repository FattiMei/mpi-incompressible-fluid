#ifndef TEST_CASE_BOUNDARIES_H
#define TEST_CASE_BOUNDARIES_H

#include "Real.h"

namespace mif {

constexpr Real exact_solution_precision = 1e-12;

// Exact solutions for the two test cases.
// Note that the solutions are only exact at time t=0,
// or at the borders where Dirichlet BC are imposed. 
// Nevertheless, they are defined everywhere, but should 
// not be used aside from those cases.

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
inline Real exact_u_t1(Real t, Real x, Real y, Real z) {
    return 0.0;
}

inline Real exact_v_t1(Real t, Real x, Real y, Real z) {
    if (x < 1.0+exact_solution_precision && x > 1.0-exact_solution_precision) {
        return 1.0;
    }
    return 0.0;
}

inline Real exact_w_t1(Real t, Real x, Real y, Real z) {
    return 0.0;
}

inline Real exact_p_initial_t1(Real x, Real y, Real z) {
    return 0.0;
}


inline Real exact_u_t2(Real t, Real x, Real y, Real z) {
    return 0.0;
}

inline Real exact_v_t2(Real t, Real x, Real y, Real z) {
    if (x < -0.5+exact_solution_precision && x > -0.5-exact_solution_precision) {
        return 1.0;
    }
    return 0.0;
}

inline Real exact_w_t2(Real t, Real x, Real y, Real z) {
    return 0.0;
}

inline Real exact_p_initial_t2(Real x, Real y, Real z) {
    return 0.0;
}
#pragma GCC diagnostic pop

} // mif

#endif // TEST_CASE_BOUNDARIES_H