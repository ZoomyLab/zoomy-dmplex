#pragma once
#include "Numerics.H"
// VAM Chorin pressure + corrector ops (generated). State: ['b', 'h', 'q_0', 'q_1', 'r_0', 'r_1', 'P_0', 'P_1'].
// P modes = state[[6, 7]]; corrector updates state[[2, 3, 4, 5]].
// Derivative-aux layout the solver fills (frozen-state h/b/q once per step;
// P-derivatives recomputed each KSP matvec):
//   Qaux[0] = P_0_x  (P_0 d/x)
//   Qaux[1] = P_0_x_x  (P_0 d/xx)
//   Qaux[2] = P_1_x  (P_1 d/x)
//   Qaux[3] = P_1_x_x  (P_1 d/xx)
//   Qaux[4] = b_x  (b d/x)
//   Qaux[5] = b_x_x  (b d/xx)
//   Qaux[6] = h_x  (h d/x)
//   Qaux[7] = h_x_x  (h d/xx)
//   Qaux[8] = q_0_x  (q_0 d/x)
//   Qaux[9] = q_1_x  (q_1 d/x)

// ── the solver contract (GENERATED — never hard-code these) ───────────────
// These change with `dimension`: dim=2 (1-D horizontal, the Escalante bump)
// gives NF=8/NCA=10; dim=3 (2-D horizontal) gives NF=10/NCA=20. The solver
// must read them from here, or a dim=2 run silently indexes a dim=3 layout.
// NS is the PREDICTOR's evolved dof (pressure dropped by the splitter) and
// must equal Model<T>::n_dof_q — static_assert'd solver-side.
static constexpr int VAM_NF  = 8;   // full VAM state (X + P)
static constexpr int VAM_NP  = 2;   // pressure modes
static constexpr int VAM_NS  = VAM_NF - VAM_NP; // evolved state dof
static constexpr int VAM_NCA = 10;  // derivative-aux entries per cell
static constexpr int VAM_P_IDX[VAM_NP] = {6, 7};
static constexpr int VAM_CORR_IDX[4] = {2, 3, 4, 5};
static constexpr int VAM_N_CORR = 4;

// CA fill table: one row per derivative aux, in CA order.
//   field = index into the FULL VAM state (Q above)
//   ax1   = first  derivative axis (0=x, 1=y, 2=z)
//   ax2   = second derivative axis, or -1 for a 1st derivative
//   is_P  = pressure aux -> recomputed EVERY KSP matvec from the current P
//           iterate; !is_P -> frozen state aux, filled once per step.
struct VamAuxSpec { int field; int ax1; int ax2; bool is_P; };
static constexpr VamAuxSpec VAM_CA[VAM_NCA] = {
    {6, 0, -1, true},   // CA[0] = P_0_x
    {6, 0, 0, true},   // CA[1] = P_0_x_x
    {7, 0, -1, true},   // CA[2] = P_1_x
    {7, 0, 0, true},   // CA[3] = P_1_x_x
    {0, 0, -1, false},   // CA[4] = b_x
    {0, 0, 0, false},   // CA[5] = b_x_x
    {1, 0, -1, false},   // CA[6] = h_x
    {1, 0, 0, false},   // CA[7] = h_x_x
    {2, 0, -1, false},   // CA[8] = q_0_x
    {3, 0, -1, false},   // CA[9] = q_1_x
};

template <typename T>
PORTABLE_FN inline void vam_pressure_residual(const T* Q, const T* Qaux, const T* p, T dt, T* out) {
    out[0] = -Q[1]*Qaux[1]*dt/p[1] + (1.0/3.0)*Q[1]*Qaux[3]*dt/p[1] + 2*Q[6]*Qaux[5]*dt/p[1] + 4*Q[7]*Qaux[5]*dt/p[1] + (2.0/3.0)*Q[7]*Qaux[7]*dt/p[1] + 4*Qaux[0]*Qaux[4]*dt/p[1] + 4*Qaux[2]*Qaux[4]*dt/p[1] + (4.0/3.0)*Qaux[2]*Qaux[6]*dt/p[1] + Qaux[8] - 1.0/3.0*Qaux[9] - 2*Q[2]*Qaux[4]/Q[1] - Q[2]*Qaux[6]/Q[1] - 1.0/3.0*Q[3]*Qaux[6]/Q[1] + 2*Q[4]/Q[1] + 4*Q[6]*Qaux[4]*Qaux[6]*dt/(Q[1]*p[1]) + 2*Q[6]*pow(Qaux[6], 2)*dt/(Q[1]*p[1]) - 4*Q[7]*pow(Qaux[4], 2)*dt/(Q[1]*p[1]) + (2.0/3.0)*Q[7]*pow(Qaux[6], 2)*dt/(Q[1]*p[1]) - 4*Q[7]*dt/(Q[1]*p[1]);
    out[1] = -Q[1]*Qaux[1]*dt/p[1] - Q[6]*Qaux[7]*dt/p[1] + 2*Q[7]*Qaux[5]*dt/p[1] - Qaux[0]*Qaux[6]*dt/p[1] + 4*Qaux[2]*Qaux[4]*dt/p[1] + Qaux[2]*Qaux[6]*dt/p[1] + Qaux[8] - Q[2]*Qaux[6]/Q[1] - 2*Q[3]*Qaux[4]/Q[1] - Q[3]*Qaux[6]/Q[1] + 2*Q[5]/Q[1] + 12*Q[6]*pow(Qaux[4], 2)*dt/(Q[1]*p[1]) + 12*Q[6]*Qaux[4]*Qaux[6]*dt/(Q[1]*p[1]) + 4*Q[6]*pow(Qaux[6], 2)*dt/(Q[1]*p[1]) + 12*Q[6]*dt/(Q[1]*p[1]) + 12*Q[7]*pow(Qaux[4], 2)*dt/(Q[1]*p[1]) + 8*Q[7]*Qaux[4]*Qaux[6]*dt/(Q[1]*p[1]) + 2*Q[7]*pow(Qaux[6], 2)*dt/(Q[1]*p[1]) + 12*Q[7]*dt/(Q[1]*p[1]);
}
template <typename T>
PORTABLE_FN inline void vam_corrector(const T* Q, const T* Qaux, const T* p, T dt, T* out) {
    out[0] = -Q[1]*Qaux[0]*dt/p[1] + Q[2] - Q[6]*Qaux[6]*dt/p[1] + 2*Q[7]*Qaux[4]*dt/p[1];
    out[1] = -Q[1]*Qaux[2]*dt/p[1] + Q[3] - 6*Q[6]*Qaux[4]*dt/p[1] - 3*Q[6]*Qaux[6]*dt/p[1] - 6*Q[7]*Qaux[4]*dt/p[1] - 2*Q[7]*Qaux[6]*dt/p[1];
    out[2] = Q[4] - 2*Q[7]*dt/p[1];
    out[3] = Q[5] + 6*Q[6]*dt/p[1] + 6*Q[7]*dt/p[1];
}
