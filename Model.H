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
    static constexpr int n_dof_q    = 3;
    static constexpr int n_dof_qaux = 4;
    static constexpr int n_parameters = 5;
    static constexpr int dimension  = 1;
    static constexpr int n_dof_gradQ = 0;
    static constexpr bool has_diffusion = false;
    static constexpr bool has_free_surface = true;
    static constexpr int idx_b = 0;
    static constexpr int idx_h = 1;
    static constexpr int n_boundary_tags = 1;
    static const std::vector<std::string> get_boundary_tags() { return { "outer" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "rho", "nu", "lambda_s", "e_x" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 1.0, 0.0, 0.0, 0.0 }; }
    PORTABLE_FN static inline SimpleArray<T, 3> flux_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = Q[2];
    res[2] = std::pow(Q[2], 2)/Q[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 9> nonconservative_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 9> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = p[0]*Q[1];
    res[7] = 0;
    res[8] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 9> quasilinear_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = p[0]*Q[1];
    SimpleArray<T, 9> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 1;
    res[6] = t0;
    res[7] = t0 - std::pow(Q[2], 2)/std::pow(Q[1], 2);
    res[8] = 2*Q[2]/Q[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = (1.0 / std::pow(Q[1], 1));
    T t1 = std::pow(p[0], 1.0/2.0)*n[0]*std::pow(std::pow(Q[1], 5), 1.0/2.0)/std::pow(Q[1], 2);
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = n[0]*Q[2]*t0 - t1;
    res[2] = n[0]*Q[2]*t0 + t1;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> source(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 2*Qaux[0]/p[1] + p[4]*p[0]*Q[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> flux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = Q[2];
    res[2] = std::pow(Q[2], 2)/Q[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 9> nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 9> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = p[0]*Q[1];
    res[7] = 0;
    res[8] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 9> source_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 9> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = p[4]*p[0];
    res[8] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 12> source_jacobian_wrt_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 12> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 2/p[1];
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
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
    res[1] = compute_derivative(Q[2], 1, 0, 0);
    res[2] = compute_derivative(Q[1], 1, 0, 0);
    res[3] = compute_derivative(Q[0], 1, 0, 0);
    return res;
    }

    // initial_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel carries no symbolic initial state; IC is
    //   case-side (case directory / settings.json).

    // initial_aux_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel carries no symbolic initial state; IC is
    //   case-side (case directory / settings.json).

    PORTABLE_FN static inline SimpleArray<T, 3> reconstruction_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 3> res;
    res[0] = Q[0];
    res[1] = Q[0] + Q[1];
    res[2] = Q[2]/Q[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> state_from_reconstruction(
        const T* W,
        const T* Qaux,
        const T* p)
    {
    T t0 = -W[0] + W[1];
    SimpleArray<T, 3> res;
    res[0] = W[0];
    res[1] = t0;
    res[2] = W[2]*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> interpolate_to_3d(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* X)
    {
    T t0 = Q[2]/Q[1];
    SimpleArray<T, 6> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = t0;
    res[3] = 0;
    res[4] = Qaux[3]*t0 + Qaux[2]*t0*X[2] - Qaux[1]*X[2];
    res[5] = -p[0]*Q[1]*p[1]*X[2] + p[0]*Q[1]*p[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> project_from_3d(
        const T* profile,
        const T* p,
        const T* I)
    {
    SimpleArray<T, 3> res;
    res[0] = profile[0];
    res[1] = profile[1];
    res[2] = profile[1]*I[0];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> boundary_conditions(
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
        SimpleArray<T, 3> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = -Q[2];
        return res;
    }
    SimpleArray<T, 3> default_res = {0};
    return default_res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> aux_boundary_conditions(
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
        res[0] = Qaux[0];
        res[1] = Qaux[1];
        res[2] = Qaux[2];
        res[3] = Qaux[3];
        return res;
    }
    SimpleArray<T, 4> default_res = {0};
    return default_res;
    }

};
