#ifndef MANUFACTURED_H
#define MANUFACTURED_H

extern double Reynolds;

// Exact solutions for the velocity+pressure tests.
double u_exact(double t, double x, double y, double z);
double v_exact(double t, double x, double y, double z);
double w_exact(double t, double x, double y, double z);
double p_exact(double t, double x, double y, double z);
double dp_dx_exact(double t, double x, double y, double z);
double dp_dy_exact(double t, double x, double y, double z);
double dp_dz_exact(double t, double x, double y, double z);

#endif // MANUFACTURED_H
