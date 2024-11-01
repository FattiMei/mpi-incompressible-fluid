#ifndef MANUFACTURED_HH
#define MANUFACTURED_HH
#include <Real.h>
#include <cmath>
#define sin sinf
#define cos cosf
#define pow(x, h) x*x
extern Real Reynolds;


const Real u_exact(Real t, Real x, Real y, Real z) noexcept;
const Real v_exact(Real t, Real x, Real y, Real z)noexcept;
const Real w_exact(Real t, Real x, Real y, Real z)noexcept;

const Real p_exact(Real t, Real x, Real y, Real z) noexcept;
const Real forcing_x(Real t, Real x, Real y, Real z) noexcept;
const Real forcing_y(Real t, Real x, Real y, Real z)noexcept;
const Real forcing_z(Real t, Real x, Real y, Real z)noexcept;

#endif // MANUFACTURED_HH
