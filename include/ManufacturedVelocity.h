#ifndef MANUFACTURED_VELOCITY_H
#define MANUFACTURED_VELOCITY_H

extern double Reynolds;

// Exact solutions for the standalone velocity tests.
double u_exact_v_test(double t, double x, double y, double z);
double v_exact_v_test(double t, double x, double y, double z);
double w_exact_v_test(double t, double x, double y, double z);
double forcing_x(double t, double x, double y, double z);
double forcing_y(double t, double x, double y, double z);
double forcing_z(double t, double x, double y, double z);

#endif // MANUFACTURED_VELOCITY_H
