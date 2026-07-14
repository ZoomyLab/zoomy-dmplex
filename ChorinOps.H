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
    out[0] = -Q[1]*Qaux[1]*dt/p[1] - Q[1]*Qaux[3]*dt/p[1] + (1.0/3.0)*Q[1]*Qaux[5]*dt/p[1] + (1.0/3.0)*Q[1]*Qaux[7]*dt/p[1] + 2*Q[8]*Qaux[11]*dt/p[1] + 2*Q[8]*Qaux[9]*dt/p[1] + 4*Q[9]*Qaux[11]*dt/p[1] + (2.0/3.0)*Q[9]*Qaux[13]*dt/p[1] + (2.0/3.0)*Q[9]*Qaux[15]*dt/p[1] + 4*Q[9]*Qaux[9]*dt/p[1] + 4*Qaux[0]*Qaux[8]*dt/p[1] + 4*Qaux[10]*Qaux[2]*dt/p[1] + 4*Qaux[10]*Qaux[6]*dt/p[1] + (4.0/3.0)*Qaux[12]*Qaux[4]*dt/p[1] + (4.0/3.0)*Qaux[14]*Qaux[6]*dt/p[1] + Qaux[16] - 1.0/3.0*Qaux[17] + Qaux[18] - 1.0/3.0*Qaux[19] + 4*Qaux[4]*Qaux[8]*dt/p[1] - Q[2]*Qaux[12]/Q[1] - 2*Q[2]*Qaux[8]/Q[1] - 1.0/3.0*Q[3]*Qaux[12]/Q[1] - 2*Q[4]*Qaux[10]/Q[1] - Q[4]*Qaux[14]/Q[1] - 1.0/3.0*Q[5]*Qaux[14]/Q[1] + 2*Q[6]/Q[1] + 4*Q[8]*Qaux[10]*Qaux[14]*dt/(Q[1]*p[1]) + 2*Q[8]*pow(Qaux[12], 2)*dt/(Q[1]*p[1]) + 4*Q[8]*Qaux[12]*Qaux[8]*dt/(Q[1]*p[1]) + 2*Q[8]*pow(Qaux[14], 2)*dt/(Q[1]*p[1]) - 4*Q[9]*pow(Qaux[10], 2)*dt/(Q[1]*p[1]) + (2.0/3.0)*Q[9]*pow(Qaux[12], 2)*dt/(Q[1]*p[1]) + (2.0/3.0)*Q[9]*pow(Qaux[14], 2)*dt/(Q[1]*p[1]) - 4*Q[9]*pow(Qaux[8], 2)*dt/(Q[1]*p[1]) - 4*Q[9]*dt/(Q[1]*p[1]);
    out[1] = -Q[1]*Qaux[1]*dt/p[1] - Q[1]*Qaux[3]*dt/p[1] - Q[8]*Qaux[13]*dt/p[1] - Q[8]*Qaux[15]*dt/p[1] + 2*Q[9]*Qaux[11]*dt/p[1] + 2*Q[9]*Qaux[9]*dt/p[1] - Qaux[0]*Qaux[12]*dt/p[1] + 4*Qaux[10]*Qaux[6]*dt/p[1] + Qaux[12]*Qaux[4]*dt/p[1] - Qaux[14]*Qaux[2]*dt/p[1] + Qaux[14]*Qaux[6]*dt/p[1] + Qaux[16] + Qaux[18] + 4*Qaux[4]*Qaux[8]*dt/p[1] - Q[2]*Qaux[12]/Q[1] - Q[3]*Qaux[12]/Q[1] - 2*Q[3]*Qaux[8]/Q[1] - Q[4]*Qaux[14]/Q[1] - 2*Q[5]*Qaux[10]/Q[1] - Q[5]*Qaux[14]/Q[1] + 2*Q[7]/Q[1] + 12*Q[8]*pow(Qaux[10], 2)*dt/(Q[1]*p[1]) + 12*Q[8]*Qaux[10]*Qaux[14]*dt/(Q[1]*p[1]) + 4*Q[8]*pow(Qaux[12], 2)*dt/(Q[1]*p[1]) + 12*Q[8]*Qaux[12]*Qaux[8]*dt/(Q[1]*p[1]) + 4*Q[8]*pow(Qaux[14], 2)*dt/(Q[1]*p[1]) + 12*Q[8]*pow(Qaux[8], 2)*dt/(Q[1]*p[1]) + 12*Q[8]*dt/(Q[1]*p[1]) + 12*Q[9]*pow(Qaux[10], 2)*dt/(Q[1]*p[1]) + 8*Q[9]*Qaux[10]*Qaux[14]*dt/(Q[1]*p[1]) + 2*Q[9]*pow(Qaux[12], 2)*dt/(Q[1]*p[1]) + 8*Q[9]*Qaux[12]*Qaux[8]*dt/(Q[1]*p[1]) + 2*Q[9]*pow(Qaux[14], 2)*dt/(Q[1]*p[1]) + 12*Q[9]*pow(Qaux[8], 2)*dt/(Q[1]*p[1]) + 12*Q[9]*dt/(Q[1]*p[1]);
}
template <typename T>
PORTABLE_FN inline void vam_corrector(const T* Q, const T* Qaux, const T* p, T dt, T* out) {
    out[0] = -Q[1]*Qaux[0]*dt/p[1] + Q[2] - Q[8]*Qaux[12]*dt/p[1] + 2*Q[9]*Qaux[8]*dt/p[1];
    out[1] = -Q[1]*Qaux[4]*dt/p[1] + Q[3] - 3*Q[8]*Qaux[12]*dt/p[1] - 6*Q[8]*Qaux[8]*dt/p[1] - 2*Q[9]*Qaux[12]*dt/p[1] - 6*Q[9]*Qaux[8]*dt/p[1];
    out[2] = -Q[1]*Qaux[2]*dt/p[1] + Q[4] - Q[8]*Qaux[14]*dt/p[1] + 2*Q[9]*Qaux[10]*dt/p[1];
    out[3] = -Q[1]*Qaux[6]*dt/p[1] + Q[5] - 6*Q[8]*Qaux[10]*dt/p[1] - 3*Q[8]*Qaux[14]*dt/p[1] - 6*Q[9]*Qaux[10]*dt/p[1] - 2*Q[9]*Qaux[14]*dt/p[1];
    out[4] = Q[6] - 2*Q[9]*dt/p[1];
    out[5] = Q[7] + 6*Q[8]*dt/p[1] + 6*Q[9]*dt/p[1];
}
