#pragma once
#include <cmath>
#include <array>

#ifndef ZOOMY_SIMPLE_ARRAY
#define ZOOMY_SIMPLE_ARRAY
template <typename T, int N>
struct SimpleArray {
    T data[N];
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
    T* begin() { return data; }
    const T* begin() const { return data; }
    T* end() { return data + N; }
    const T* end() const { return data + N; }
};
#endif

#include <vector>
#include <string>
#include <algorithm>

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif

template <typename T>
struct Model {
    static constexpr int n_dof_q    = 6;
    static constexpr int n_dof_qaux = 4;
    static constexpr int n_parameters = 4;
    static constexpr int dimension  = 1;
    static constexpr int n_dof_gradQ = 0;
    static constexpr bool has_diffusion = false;
    static constexpr bool has_free_surface = true;
    static constexpr int idx_b = 0;
    static constexpr int idx_h = 1;
    static constexpr int n_boundary_tags = 2;
    static const std::vector<std::string> get_boundary_tags() { return { "left", "right" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "rho", "nu", "lambda_s" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 1.0, 0.0, 0.0 }; }
    PORTABLE_FN static inline SimpleArray<T, 6> flux_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Q[3], 2);
    T t1 = (1.0/3.0)*Qaux[3];
    T t2 = Qaux[3]*Q[2];
    T t3 = Q[3]*t2;
    T t4 = Q[3]*Q[5];
    T t5 = (2.0/5.0)*Qaux[3];
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = Q[2];
    res[2] = Qaux[3]*std::pow(Q[2], 2) + t0*t1;
    res[3] = 2*t3;
    res[4] = Q[4]*t2 + t1*t4;
    res[5] = -Qaux[0]*t0*t5 + (2.0/5.0)*Qaux[0]*t3 + (3.0/5.0)*Qaux[3]*Q[3]*Q[4] + Q[5]*t2 + t4*t5;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 48> nonconservative_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 48> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = p[0]*Q[1];
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = -Qaux[3]*Q[2];
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = 0;
    res[42] = 0;
    res[43] = -6.0/5.0*Qaux[3]*Q[4] + (1.0/5.0)*Qaux[3]*Q[5];
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 48> quasilinear_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[3], 2);
    T t1 = std::pow(Q[3], 2);
    T t2 = (1.0/3.0)*t0;
    T t3 = Qaux[3]*Q[2];
    T t4 = Qaux[3]*Q[3];
    T t5 = Q[2]*t0;
    T t6 = Q[3]*t5;
    T t7 = Q[3]*Q[5];
    T t8 = Qaux[3]*Q[4];
    T t9 = Qaux[3]*Q[5];
    T t10 = (2.0/5.0)*Qaux[0];
    T t11 = (2.0/5.0)*t4;
    SimpleArray<T, 48> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = 0;
    res[10] = 1;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = p[0]*Q[1];
    res[17] = p[0]*Q[1] - std::pow(Q[2], 2)*t0 - t1*t2;
    res[18] = 2*t3;
    res[19] = (2.0/3.0)*t4;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = -2*t6;
    res[26] = 2*t4;
    res[27] = t3;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = -Q[4]*t5 - t2*t7;
    res[34] = t8;
    res[35] = (1.0/3.0)*t9;
    res[36] = t3;
    res[37] = (1.0/3.0)*t4;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = (2.0/5.0)*Qaux[0]*t0*t1 - 3.0/5.0*Q[3]*Q[4]*t0 - Q[5]*t5 - 2.0/5.0*t0*t7 - t10*t6;
    res[42] = Qaux[0]*t11 + t9;
    res[43] = -4.0/5.0*Qaux[0]*t4 + t10*t3 - 3.0/5.0*t8 + (3.0/5.0)*t9;
    res[44] = (3.0/5.0)*t4;
    res[45] = t11 + t3;
    res[46] = 0;
    res[47] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> source(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[3], 2)*p[2]*Q[3];
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = -2*t0;
    res[3] = -6*t0;
    res[4] = 0;
    res[5] = -1.0/5.0*Qaux[0]*Qaux[3]*Q[2]*Qaux[1] + (1.0/5.0)*Qaux[0]*Qaux[3]*Q[3]*Qaux[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> flux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Q[3], 2);
    T t1 = (1.0/3.0)*Qaux[3];
    T t2 = Qaux[3]*Q[2];
    T t3 = Q[3]*t2;
    T t4 = Q[3]*Q[5];
    T t5 = (2.0/5.0)*Qaux[3];
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = Q[2];
    res[2] = Qaux[3]*std::pow(Q[2], 2) + t0*t1;
    res[3] = 2*t3;
    res[4] = Q[4]*t2 + t1*t4;
    res[5] = -Qaux[0]*t0*t5 + (2.0/5.0)*Qaux[0]*t3 + (3.0/5.0)*Qaux[3]*Q[3]*Q[4] + Q[5]*t2 + t4*t5;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 48> nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 48> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = p[0]*Q[1];
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = -Qaux[3]*Q[2];
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = 0;
    res[42] = 0;
    res[43] = -6.0/5.0*Qaux[3]*Q[4] + (1.0/5.0)*Qaux[3]*Q[5];
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 48> source_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[3], 3)*p[2]*Q[3];
    T t1 = std::pow(Qaux[3], 2);
    T t2 = p[2]*t1;
    T t3 = (1.0/5.0)*Qaux[0]*Qaux[1];
    T t4 = t1*t3;
    T t5 = Qaux[3]*t3;
    SimpleArray<T, 48> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = 0;
    res[17] = 4*t0;
    res[18] = 0;
    res[19] = -2*t2;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 12*t0;
    res[26] = 0;
    res[27] = -6*t2;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = Q[2]*t4 - Q[3]*t4;
    res[42] = -t5;
    res[43] = t5;
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 24> source_jacobian_wrt_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = (1.0/5.0)*Qaux[3];
    SimpleArray<T, 24> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = 0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = (1.0/5.0)*Qaux[3]*Q[3]*Qaux[1] - Q[2]*Qaux[1]*t0;
    res[21] = (1.0/5.0)*Qaux[0]*Qaux[3]*Q[3] - Qaux[0]*Q[2]*t0;
    res[22] = 0;
    res[23] = 0;
    return res;
    }

    // update_variables: NOT emitted from the SystemModel (provide solver-side).
    //   no pointwise state remap on this SystemModel; the solver
    //   leaves Q unchanged after the transport step.

    PORTABLE_FN static inline SimpleArray<T, 4> update_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T dt)
    {
    SimpleArray<T, 4> res;
    res[0] = Qaux[0];
    res[1] = Qaux[1];
    res[2] = Qaux[2];
    res[3] = std::pow((T)2, 1.0/2.0)*std::max((T)0, Q[1])/std::pow(std::pow(Q[1], 4) + std::pow(std::max((T)1.0e-8, Q[1]), 4), 1.0/2.0);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 32> update_aux_variables_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T dt)
    {
    T t0 = std::pow((T)2, 1.0/2.0);
    T t1 = std::max((T)1.0e-8, Q[1]);
    T t2 = std::pow(Q[1], 4) + std::pow(t1, 4);
    SimpleArray<T, 32> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = t0*Heaviside(Q[1], 1.0/2.0)/std::pow(t2, 1.0/2.0) + t0*(-2*std::pow(Q[1], 3) - 2*std::pow(t1, 3)*Heaviside(Q[1] - 1.0e-8, 1.0/2.0))*std::max((T)0, Q[1])/std::pow(t2, 3.0/2.0);
    res[8] = 0;
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = 0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    return res;
    }

    static inline void update_aux_variables(
        T* const* Q,
        T* const* Qaux,
        const T dt,
        const ZoomyMesh& mesh)
    {
    // Qaux[0] (b_x) = D^(1,) Q[0]
    compute_derivative(Qaux[0], Q[0], 1, 0, 0, mesh);
    // Qaux[1] (q_1_x) = D^(1,) Q[3]
    compute_derivative(Qaux[1], Q[3], 1, 0, 0, mesh);
    }

    // initial_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel's IC is trivial / zero; set it case-side.

    // initial_aux_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel's IC is trivial / zero; set it case-side.

    PORTABLE_FN static inline SimpleArray<T, 8> boundary_conditions(
        const int bc_idx,
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n,
        const T* X,
        const T time,
        const T dX)
    {
    if (bc_idx == 0) {
        SimpleArray<T, 8> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = 0.11197;
        res[3] = 0.0;
        res[4] = 0.0;
        res[5] = 0.0;
        res[6] = Q[6];
        res[7] = Q[7];
        return res;
    } else if (bc_idx == 1) {
        SimpleArray<T, 8> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = Q[2];
        res[3] = Q[3];
        res[4] = Q[4];
        res[5] = Q[5];
        res[6] = 0.0;
        res[7] = 0.0;
        return res;
    }
    SimpleArray<T, 8> default_res = {0};
    return default_res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> aux_boundary_conditions(
        const int bc_idx,
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n,
        const T* X,
        const T time,
        const T dX)
    {
    if (bc_idx == 0 || bc_idx == 1) {
        SimpleArray<T, 3> res;
        res[0] = Qaux[0];
        res[1] = Qaux[1];
        res[2] = Qaux[2];
        return res;
    }
    SimpleArray<T, 3> default_res = {0};
    return default_res;
    }

};

// generated by create_vam_model.py (REQ-169): selects the real
// ChorinVAMSolver. create_model.py does NOT emit this, so a non-VAM
// tree gets the loud runtime stub instead of a compile error.
#define ZOOMY_MODEL_IS_VAM 1
