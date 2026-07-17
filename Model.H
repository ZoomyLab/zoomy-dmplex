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
    static constexpr double dt_max = 5.0;
    static constexpr int n_dof_gradQ = 0;
    static constexpr bool has_diffusion = false;
    static constexpr bool has_free_surface = true;
    static constexpr int idx_b = 0;
    static constexpr int idx_h = 1;
    static constexpr int n_boundary_tags = 1;
    static const std::vector<std::string> get_boundary_tags() { return { "default" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "rho", "nu", "lambda_s", "e_x" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 1.0, 1.0, 0.5, 0.0 }; }
    PORTABLE_FN static inline SimpleArray<T, 3> flux_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = Q[2];
    res[2] = Qaux[3]*std::pow(Q[2], 2);
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
    res[7] = -std::pow(Qaux[3], 2)*std::pow(Q[2], 2) + t0;
    res[8] = 2*Qaux[3]*Q[2];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = std::max((T)1.0e-14, Q[1]);
    T t1 = (1.0 / std::pow(t0, 1));
    T t2 = std::pow(p[0], 1.0/2.0)*n[0]*std::pow(std::max((T)0, std::pow(Q[1], 5)), 1.0/2.0)/std::pow(t0, 2);
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = n[0]*Q[2]*t1 - t2;
    res[2] = n[0]*Q[2]*t1 + t2;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> source(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T time,
        const T dt,
        const T* X)
    {
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = p[4]*p[0]*Q[1] - Qaux[3]*p[3]*Q[2]/p[1];
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
    res[2] = Qaux[3]*std::pow(Q[2], 2);
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
    T t0 = p[3]/p[1];
    SimpleArray<T, 9> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = p[4]*p[0] + std::pow(Qaux[3], 2)*Q[2]*t0;
    res[8] = -Qaux[3]*t0;
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
    res[8] = 0;
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> update_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T dt)
    {
    T t0 = 30.0*std::max((T)0, Q[1] - 0.01);
    SimpleArray<T, 3> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = std::max(-t0, std::min(Q[2], t0));
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> update_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T time,
        const T* X)
    {
    SimpleArray<T, 4> res;
    res[0] = Qaux[0];
    res[1] = Qaux[1];
    res[2] = Qaux[2];
    res[3] = std::pow((T)2, 1.0/2.0)*std::max((T)0, Q[1])/std::pow(std::pow(Q[1], 4) + std::pow(std::max((T)1.0e-8, Q[1]), 4), 1.0/2.0);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 12> update_aux_variables_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T time,
        const T* X)
    {
    T t0 = std::pow((T)2, 1.0/2.0);
    T t1 = std::max((T)1.0e-8, Q[1]);
    T t2 = std::pow(Q[1], 4) + std::pow(t1, 4);
    SimpleArray<T, 12> res;
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
    return res;
    }

    static inline void update_aux_variables(
        T* const* Q,
        T* const* Qaux,
        const T dt,
        const ZoomyMesh& mesh)
    {
    // Qaux[0] (dq0dx) = D^(1,) Q[2]
    compute_derivative(Qaux[0], Q[2], 1, 0, 0, mesh);
    // Qaux[1] (dhdx) = D^(1,) Q[1]
    compute_derivative(Qaux[1], Q[1], 1, 0, 0, mesh);
    // Qaux[2] (dbdx) = D^(1,) Q[0]
    compute_derivative(Qaux[2], Q[0], 1, 0, 0, mesh);
    }

    PORTABLE_FN static inline SimpleArray<T, 3> initial_condition(
        const T* X,
        const T* p)
    {
    SimpleArray<T, 3> res;
    res[0] = 1.0;
    res[1] = 0.0;
    res[2] = 0.0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> initial_aux_condition(
        const T* X,
        const T* p)
    {
    SimpleArray<T, 4> res;
    res[0] = 1.0;
    res[1] = 0.0;
    res[2] = 0.0;
    res[3] = 0.0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> reconstruction_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 3> res;
    res[0] = Q[0];
    res[1] = Q[0] + Q[1];
    res[2] = Qaux[3]*Q[2];
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
    res[4] = Qaux[2]*t0 + Qaux[1]*t0*X[2] - Qaux[0]*X[2];
    res[5] = -p[0]*Q[1]*p[1]*X[2] + p[0]*Q[1]*p[1];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 3> project_from_3d(
        const T* const* column,
        const T* p)
    {
    T t0 = 0.030366692131398035*column[0][1];
    T t1 = 0.030354755538579095*column[0][1];
    T t2 = 0.030281980827521641*column[0][1];
    T t3 = 0.030277745262327821*column[0][1];
    T t4 = 0.030274279799896515*column[0][1];
    T t5 = 0.030271584440227723*column[0][1];
    T t6 = 0.030269659183321442*column[0][1];
    T t7 = 0.030268504029177669*column[0][1];
    T t8 = 0.030343589048522655*column[0][1];
    T t9 = 0.030286986495477976*column[0][1];
    T t10 = 0.030292762266196819*column[0][1];
    T t11 = 0.030299308139678176*column[0][1];
    T t12 = 0.030306624115922047*column[0][1];
    T t13 = 0.030314710194928436*column[0][1];
    T t14 = 0.030323566376697326*column[0][1];
    T t15 = 0.030333192661228733*column[0][1];
    SimpleArray<T, 3> res;
    res[0] = column[0][0];
    res[1] = column[0][1];
    res[2] = 0.030268118977796417*column[0][1]*column[16][2] + column[0][2]*t0 + column[1][2]*t1 + column[10][2]*t2 + column[11][2]*t3 + column[12][2]*t4 + column[13][2]*t5 + column[14][2]*t6 + column[15][2]*t7 + column[17][2]*t7 + column[18][2]*t6 + column[19][2]*t5 + column[2][2]*t8 + column[20][2]*t4 + column[21][2]*t3 + column[22][2]*t2 + column[23][2]*t9 + column[24][2]*t10 + column[25][2]*t11 + column[26][2]*t12 + column[27][2]*t13 + column[28][2]*t14 + column[29][2]*t15 + column[3][2]*t15 + column[30][2]*t8 + column[31][2]*t1 + column[32][2]*t0 + column[4][2]*t14 + column[5][2]*t13 + column[6][2]*t12 + column[7][2]*t11 + column[8][2]*t10 + column[9][2]*t9;
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
    if (bc_idx == 0) {
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
