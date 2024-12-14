#ifndef MANUFACTURED_H
#define MANUFACTURED_H

extern double Reynolds;

double u_exact(double t, double x, double y, double z);
double v_exact(double t, double x, double y, double z);
double w_exact(double t, double x, double y, double z);
double p_exact(double t, double x, double y, double z);

#endif // MANUFACTURED_H
