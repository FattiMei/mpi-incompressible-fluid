#ifndef MANUFACTURED__H
#define MANUFACTURED__H

extern Real Reynolds;

Real u_exact(Real t, Real x, Real y, Real z);

Real v_exact(Real t, Real x, Real y, Real z);

Real w_exact(Real t, Real x, Real y, Real z);

Real p_exact(Real t, Real x, Real y, Real z);

Real forcing_x(Real t, Real x, Real y, Real z);

Real forcing_y(Real t, Real x, Real y, Real z);

Real forcing_z(Real t, Real x, Real y, Real z);

#endif // MANUFACTURED__H
