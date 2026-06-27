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
    static constexpr int n_dof_q    = 4;
    static constexpr int n_dof_qaux = 8;
    static constexpr int n_parameters = 5;
    static constexpr int dimension  = 2;
    static constexpr int n_dof_gradQ = 0;
    static constexpr bool has_diffusion = false;
    static constexpr bool has_free_surface = true;
    static constexpr int idx_b = 0;
    static constexpr int idx_h = 1;
    static constexpr int n_boundary_tags = 1;
    static const std::vector<std::string> get_boundary_tags() { return { "default" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "rho", "nu", "lambda_s", "e_x" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 1.0, 0.0, 0.0, 0.0 }; }
    PORTABLE_FN static inline SimpleArray<T, 4> flux_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = (1.0 / std::pow(Q[1], 1));
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = Q[2];
    res[2] = std::pow(Q[2], 2)*t0;
    res[3] = Q[2]*Q[3]*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> flux_y(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = (1.0 / std::pow(Q[1], 1));
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = Q[3];
    res[2] = Q[2]*Q[3]*t0;
    res[3] = std::pow(Q[3], 2)*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> nonconservative_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 16> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = p[0]*Q[1];
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> nonconservative_matrix_y(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 16> res;
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
    res[12] = p[0]*Q[1];
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> quasilinear_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = p[0]*Q[1];
    T t1 = (1.0 / std::pow(Q[1], 2));
    T t2 = (1.0 / std::pow(Q[1], 1));
    T t3 = Q[2]*t2;
    SimpleArray<T, 16> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 1;
    res[7] = 0;
    res[8] = t0;
    res[9] = -std::pow(Q[2], 2)*t1 + t0;
    res[10] = 2*t3;
    res[11] = 0;
    res[12] = 0;
    res[13] = -Q[2]*Q[3]*t1;
    res[14] = Q[3]*t2;
    res[15] = t3;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> quasilinear_matrix_y(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = (1.0 / std::pow(Q[1], 2));
    T t1 = (1.0 / std::pow(Q[1], 1));
    T t2 = Q[3]*t1;
    T t3 = p[0]*Q[1];
    SimpleArray<T, 16> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 1;
    res[8] = 0;
    res[9] = -Q[2]*Q[3]*t0;
    res[10] = t2;
    res[11] = Q[2]*t1;
    res[12] = t3;
    res[13] = -std::pow(Q[3], 2)*t0 + t3;
    res[14] = 0;
    res[15] = 2*t2;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = n[0]*Q[2];
    T t1 = n[1]*Q[3];
    T t2 = (1.0 / std::pow(Q[1], 2));
    T t3 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0)*std::pow(std::pow(Q[1], 5), 1.0/2.0);
    T t4 = Q[1]*t0 + Q[1]*t1;
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = (t0 + t1)/Q[1];
    res[2] = t2*(t3 + t4);
    res[3] = t2*(-t3 + t4);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> source(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = 2/p[1];
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = Qaux[0]*t0 + p[4]*p[0]*Q[1];
    res[3] = Qaux[1]*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 8> flux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = (1.0 / std::pow(Q[1], 1));
    T t1 = Q[2]*Q[3]*t0;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = Q[2];
    res[3] = Q[3];
    res[4] = std::pow(Q[2], 2)*t0;
    res[5] = t1;
    res[6] = t1;
    res[7] = std::pow(Q[3], 2)*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 32> nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = p[0]*Q[1];
    SimpleArray<T, 32> res;
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
    res[16] = t0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = t0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> source_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 16> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = p[4]*p[0];
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 32> source_jacobian_wrt_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = 2/p[1];
    SimpleArray<T, 32> res;
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
    res[16] = t0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = t0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    return res;
    }

    // update_variables: NOT emitted from the SystemModel (provide solver-side).
    //   no pointwise state remap on this SystemModel; the solver
    //   leaves Q unchanged after the transport step.

    PORTABLE_FN static inline SimpleArray<T, 8> update_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T dt)
    {
    SimpleArray<T, 8> res;
    res[0] = Qaux[0];
    res[1] = Qaux[1];
    res[2] = compute_derivative(Q[2], 1, 0, 0);
    res[3] = compute_derivative(Q[1], 1, 0, 0);
    res[4] = compute_derivative(Q[3], 0, 1, 0);
    res[5] = compute_derivative(Q[1], 0, 1, 0);
    res[6] = compute_derivative(Q[0], 1, 0, 0);
    res[7] = compute_derivative(Q[0], 0, 1, 0);
    return res;
    }

    // initial_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel carries no symbolic initial state; IC is
    //   case-side (case directory / settings.json).

    // initial_aux_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel carries no symbolic initial state; IC is
    //   case-side (case directory / settings.json).

    PORTABLE_FN static inline SimpleArray<T, 4> reconstruction_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = (1.0 / std::pow(Q[1], 1));
    SimpleArray<T, 4> res;
    res[0] = Q[0];
    res[1] = Q[0] + Q[1];
    res[2] = Q[2]*t0;
    res[3] = Q[3]*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> state_from_reconstruction(
        const T* W,
        const T* Qaux,
        const T* p)
    {
    T t0 = -W[0] + W[1];
    SimpleArray<T, 4> res;
    res[0] = W[0];
    res[1] = t0;
    res[2] = W[2]*t0;
    res[3] = W[3]*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> interpolate_to_3d(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* X)
    {
    T t0 = (1.0 / std::pow(Q[1], 1));
    T t1 = Q[2]*t0;
    T t2 = Q[3]*t0;
    SimpleArray<T, 6> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = t1;
    res[3] = t2;
    res[4] = Qaux[6]*t1 + Qaux[7]*t2 + Qaux[3]*t1*X[2] + Qaux[5]*t2*X[2] - Qaux[2]*X[2] - Qaux[4]*X[2];
    res[5] = -p[0]*Q[1]*p[1]*X[2] + p[0]*Q[1]*p[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> project_from_3d(
        const T* profile,
        const T* p,
        const T* I)
    {
    SimpleArray<T, 4> res;
    res[0] = profile[0];
    res[1] = profile[1];
    res[2] = profile[1]*I[0];
    res[3] = profile[1]*I[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> boundary_conditions(
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
        SimpleArray<T, 4> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = -Q[2];
        res[3] = -Q[3];
        return res;
    }
    SimpleArray<T, 4> default_res = {0};
    return default_res;
    }

    PORTABLE_FN static inline SimpleArray<T, 8> aux_boundary_conditions(
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
        res[0] = Qaux[0];
        res[1] = Qaux[1];
        res[2] = Qaux[2];
        res[3] = Qaux[3];
        res[4] = Qaux[4];
        res[5] = Qaux[5];
        res[6] = Qaux[6];
        res[7] = Qaux[7];
        return res;
    }
    SimpleArray<T, 8> default_res = {0};
    return default_res;
    }

};
