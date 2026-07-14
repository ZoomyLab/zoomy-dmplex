#pragma once
#include "Numerics.H"
// VAM Chorin pressure + corrector ops (generated). State: ['b', 'h', 'q_x_0', 'q_x_1', 'q_y_0', 'q_y_1', 'r_0', 'r_1', 'P_0', 'P_1'].
// P modes = state[[8, 9]]; corrector updates state[[2, 3, 4, 5, 6, 7]].
// Derivative-aux layout the solver fills (frozen-state h/b/q once per step;
// P-derivatives recomputed each KSP matvec):
//   Qaux[0] = P_0_x  (P_0 d/x)
//   Qaux[1] = P_0_x_x  (P_0 d/xx)
//   Qaux[2] = P_0_y  (P_0 d/y)
//   Qaux[3] = P_0_y_y  (P_0 d/yy)
//   Qaux[4] = P_1_x  (P_1 d/x)
//   Qaux[5] = P_1_x_x  (P_1 d/xx)
//   Qaux[6] = P_1_y  (P_1 d/y)
//   Qaux[7] = P_1_y_y  (P_1 d/yy)
//   Qaux[8] = b_x  (b d/x)
//   Qaux[9] = b_x_x  (b d/xx)
//   Qaux[10] = b_y  (b d/y)
//   Qaux[11] = b_y_y  (b d/yy)
//   Qaux[12] = h_x  (h d/x)
//   Qaux[13] = h_x_x  (h d/xx)
//   Qaux[14] = h_y  (h d/y)
//   Qaux[15] = h_y_y  (h d/yy)
//   Qaux[16] = q_x_0_x  (q_x_0 d/x)
//   Qaux[17] = q_x_1_x  (q_x_1 d/x)
//   Qaux[18] = q_y_0_y  (q_y_0 d/y)
//   Qaux[19] = q_y_1_y  (q_y_1 d/y)
template <typename T>
PORTABLE_FN inline void vam_pressure_residual(const T* Q, const T* Qaux, const T* p, T dt, T* out) {
    out[0] = 4*P_0*b_x*dt*h_x/(h*rho) + 2*P_0*b_x_x*dt/rho + 4*P_0*b_y*dt*h_y/(h*rho) + 2*P_0*b_y_y*dt/rho + 2*P_0*dt*pow(h_x, 2)/(h*rho) + 2*P_0*dt*pow(h_y, 2)/(h*rho) + 4*P_0_x*b_x*dt/rho - P_0_x_x*dt*h/rho + 4*P_0_y*b_y*dt/rho - P_0_y_y*dt*h/rho - 4*P_1*pow(b_x, 2)*dt/(h*rho) + 4*P_1*b_x_x*dt/rho - 4*P_1*pow(b_y, 2)*dt/(h*rho) + 4*P_1*b_y_y*dt/rho + (2.0/3.0)*P_1*dt*h_x_x/rho + (2.0/3.0)*P_1*dt*h_y_y/rho + (2.0/3.0)*P_1*dt*pow(h_x, 2)/(h*rho) + (2.0/3.0)*P_1*dt*pow(h_y, 2)/(h*rho) - 4*P_1*dt/(h*rho) + 4*P_1_x*b_x*dt/rho + (4.0/3.0)*P_1_x*dt*h_x/rho + (1.0/3.0)*P_1_x_x*dt*h/rho + 4*P_1_y*b_y*dt/rho + (4.0/3.0)*P_1_y*dt*h_y/rho + (1.0/3.0)*P_1_y_y*dt*h/rho - 2*b_x*q_x_0/h - 2*b_y*q_y_0/h + q_x_0_x - 1.0/3.0*q_x_1_x + q_y_0_y - 1.0/3.0*q_y_1_y - h_x*q_x_0/h - 1.0/3.0*h_x*q_x_1/h - h_y*q_y_0/h - 1.0/3.0*h_y*q_y_1/h + 2*r_0/h;
    out[1] = 12*P_0*pow(b_x, 2)*dt/(h*rho) + 12*P_0*b_x*dt*h_x/(h*rho) + 12*P_0*pow(b_y, 2)*dt/(h*rho) + 12*P_0*b_y*dt*h_y/(h*rho) - P_0*dt*h_x_x/rho - P_0*dt*h_y_y/rho + 4*P_0*dt*pow(h_x, 2)/(h*rho) + 4*P_0*dt*pow(h_y, 2)/(h*rho) + 12*P_0*dt/(h*rho) - P_0_x*dt*h_x/rho - P_0_x_x*dt*h/rho - P_0_y*dt*h_y/rho - P_0_y_y*dt*h/rho + 12*P_1*pow(b_x, 2)*dt/(h*rho) + 8*P_1*b_x*dt*h_x/(h*rho) + 2*P_1*b_x_x*dt/rho + 12*P_1*pow(b_y, 2)*dt/(h*rho) + 8*P_1*b_y*dt*h_y/(h*rho) + 2*P_1*b_y_y*dt/rho + 2*P_1*dt*pow(h_x, 2)/(h*rho) + 2*P_1*dt*pow(h_y, 2)/(h*rho) + 12*P_1*dt/(h*rho) + 4*P_1_x*b_x*dt/rho + P_1_x*dt*h_x/rho + 4*P_1_y*b_y*dt/rho + P_1_y*dt*h_y/rho - 2*b_x*q_x_1/h - 2*b_y*q_y_1/h + q_x_0_x + q_y_0_y - h_x*q_x_0/h - h_x*q_x_1/h - h_y*q_y_0/h - h_y*q_y_1/h + 2*r_1/h;
}
template <typename T>
PORTABLE_FN inline void vam_corrector(const T* Q, const T* Qaux, const T* p, T dt, T* out) {
    out[0] = -P_0*dt*h_x/rho - P_0_x*dt*h/rho + 2*P_1*b_x*dt/rho + q_x_0;
    out[1] = -6*P_0*b_x*dt/rho - 3*P_0*dt*h_x/rho - 6*P_1*b_x*dt/rho - 2*P_1*dt*h_x/rho - P_1_x*dt*h/rho + q_x_1;
    out[2] = -P_0*dt*h_y/rho - P_0_y*dt*h/rho + 2*P_1*b_y*dt/rho + q_y_0;
    out[3] = -6*P_0*b_y*dt/rho - 3*P_0*dt*h_y/rho - 6*P_1*b_y*dt/rho - 2*P_1*dt*h_y/rho - P_1_y*dt*h/rho + q_y_1;
    out[4] = -2*P_1*dt/rho + r_0;
    out[5] = 6*P_0*dt/rho + 6*P_1*dt/rho + r_1;
}
