#ifndef MANUFACTURED__H
#define MANUFACTURED__H

namespace mif {
    
    extern const double Reynolds;

    double u_exact  (double t, double x, double y, double z);
    double v_exact  (double t, double x, double y, double z);
    double w_exact  (double t, double x, double y, double z);
    double p_exact  (double t, double x, double y, double z);
    double forcing_x(double t, double x, double y, double z);
    double forcing_y(double t, double x, double y, double z);
    double forcing_z(double t, double x, double y, double z);

} // mif

#endif // MANUFACTURED__H
