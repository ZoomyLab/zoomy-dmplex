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
    static constexpr int n_dof_q    = 8;
    static constexpr int n_dof_qaux = 7;
    static constexpr int n_parameters = 4;
    static constexpr int dimension  = 2;
    static constexpr int n_dof_gradQ = 0;
    static constexpr bool has_diffusion = false;
    static constexpr bool has_free_surface = true;
    static constexpr int idx_b = 0;
    static constexpr int idx_h = 1;
    static constexpr int n_boundary_tags = 4;
    static const std::vector<std::string> get_boundary_tags() { return { "bottom", "left", "right", "top" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "rho", "nu", "lambda_s" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 1.0, 0.0, 0.0 }; }
    PORTABLE_FN static inline SimpleArray<T, 8> flux_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Q[3], 2);
    T t1 = (1.0/3.0)*Qaux[6];
    T t2 = Qaux[6]*Q[2];
    T t3 = Q[3]*t2;
    T t4 = Q[3]*t1;
    T t5 = Qaux[6]*Q[3];
    T t6 = Q[4]*t5;
    T t7 = (2.0/5.0)*t5;
    T t8 = (2.0/5.0)*Qaux[0];
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = Q[2];
    res[2] = Qaux[6]*std::pow(Q[2], 2) + t0*t1;
    res[3] = 2*t3;
    res[4] = Q[4]*t2 + Q[5]*t4;
    res[5] = Q[5]*t2 + t6;
    res[6] = Q[6]*t2 + Q[7]*t4;
    res[7] = -Qaux[1]*Q[5]*t7 + (2.0/5.0)*Qaux[1]*t6 - Qaux[6]*t0*t8 + (3.0/5.0)*Q[6]*t5 + Q[7]*t2 + Q[7]*t7 + t3*t8;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 8> flux_y(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Qaux[6]*Q[2];
    T t1 = (1.0/3.0)*Qaux[6];
    T t2 = Q[5]*t1;
    T t3 = Q[5]*t0;
    T t4 = Qaux[6]*Q[4];
    T t5 = std::pow(Q[5], 2);
    T t6 = Q[5]*t4;
    T t7 = (2.0/5.0)*Qaux[6];
    T t8 = Q[5]*t7;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = Q[4];
    res[2] = Q[3]*t2 + Q[4]*t0;
    res[3] = Q[3]*t4 + t3;
    res[4] = Qaux[6]*std::pow(Q[4], 2) + t1*t5;
    res[5] = 2*t6;
    res[6] = Q[6]*t4 + Q[7]*t2;
    res[7] = -Qaux[0]*Q[3]*t8 + (2.0/5.0)*Qaux[0]*t3 - Qaux[1]*t5*t7 + (2.0/5.0)*Qaux[1]*t6 + (3.0/5.0)*Qaux[6]*Q[5]*Q[6] + Q[7]*t4 + Q[7]*t8;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 80> nonconservative_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 80> res;
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
    res[20] = p[0]*Q[1];
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
    res[32] = 0;
    res[33] = -Qaux[6]*Q[2];
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = 0;
    res[42] = 0;
    res[43] = 0;
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = 0;
    res[50] = 0;
    res[51] = 0;
    res[52] = 0;
    res[53] = -Qaux[6]*Q[4];
    res[54] = 0;
    res[55] = 0;
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = 0;
    res[62] = 0;
    res[63] = 0;
    res[64] = 0;
    res[65] = 0;
    res[66] = 0;
    res[67] = 0;
    res[68] = 0;
    res[69] = 0;
    res[70] = 0;
    res[71] = 0;
    res[72] = 0;
    res[73] = -6.0/5.0*Qaux[6]*Q[6] + (1.0/5.0)*Qaux[6]*Q[7];
    res[74] = 0;
    res[75] = 0;
    res[76] = 0;
    res[77] = 0;
    res[78] = 0;
    res[79] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 80> nonconservative_matrix_y(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 80> res;
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
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = -Qaux[6]*Q[2];
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = p[0]*Q[1];
    res[41] = 0;
    res[42] = 0;
    res[43] = 0;
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = 0;
    res[50] = 0;
    res[51] = 0;
    res[52] = 0;
    res[53] = 0;
    res[54] = 0;
    res[55] = -Qaux[6]*Q[4];
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = 0;
    res[62] = 0;
    res[63] = 0;
    res[64] = 0;
    res[65] = 0;
    res[66] = 0;
    res[67] = 0;
    res[68] = 0;
    res[69] = 0;
    res[70] = 0;
    res[71] = 0;
    res[72] = 0;
    res[73] = 0;
    res[74] = 0;
    res[75] = -6.0/5.0*Qaux[6]*Q[6] + (1.0/5.0)*Qaux[6]*Q[7];
    res[76] = 0;
    res[77] = 0;
    res[78] = 0;
    res[79] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 80> quasilinear_matrix_x(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[6], 2);
    T t1 = std::pow(Q[3], 2);
    T t2 = (1.0/3.0)*t0;
    T t3 = Qaux[6]*Q[2];
    T t4 = Qaux[6]*Q[3];
    T t5 = Q[2]*t0;
    T t6 = Q[3]*t5;
    T t7 = Q[3]*t2;
    T t8 = Qaux[6]*Q[4];
    T t9 = Qaux[6]*Q[5];
    T t10 = (1.0/3.0)*t4;
    T t11 = Q[3]*t0;
    T t12 = Q[4]*t11;
    T t13 = Qaux[6]*Q[6];
    T t14 = Qaux[6]*Q[7];
    T t15 = (2.0/5.0)*Qaux[0];
    T t16 = (2.0/5.0)*Qaux[1];
    T t17 = (2.0/5.0)*t4;
    T t18 = Qaux[1]*t17;
    SimpleArray<T, 80> res;
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
    res[12] = 1;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = 0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = p[0]*Q[1];
    res[21] = p[0]*Q[1] - std::pow(Q[2], 2)*t0 - t1*t2;
    res[22] = 2*t3;
    res[23] = (2.0/3.0)*t4;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = -2*t6;
    res[32] = 2*t4;
    res[33] = t3;
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = -Q[4]*t5 - Q[5]*t7;
    res[42] = t8;
    res[43] = (1.0/3.0)*t9;
    res[44] = t3;
    res[45] = t10;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = 0;
    res[50] = 0;
    res[51] = -Q[5]*t5 - t12;
    res[52] = t9;
    res[53] = 0;
    res[54] = t4;
    res[55] = t3;
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = -Q[6]*t5 - Q[7]*t7;
    res[62] = t13;
    res[63] = (1.0/3.0)*t14;
    res[64] = 0;
    res[65] = 0;
    res[66] = t3;
    res[67] = t10;
    res[68] = 0;
    res[69] = 0;
    res[70] = 0;
    res[71] = (2.0/5.0)*Qaux[0]*t0*t1 + (2.0/5.0)*Qaux[1]*Q[3]*Q[5]*t0 - 3.0/5.0*Q[6]*t11 - 2.0/5.0*Q[7]*t11 - Q[7]*t5 - t12*t16 - t15*t6;
    res[72] = Qaux[0]*t17 + t14;
    res[73] = -4.0/5.0*Qaux[0]*t4 - 3.0/5.0*t13 + (3.0/5.0)*t14 + t15*t3 + t16*t8 - t16*t9;
    res[74] = t18;
    res[75] = -t18;
    res[76] = (3.0/5.0)*t4;
    res[77] = t17 + t3;
    res[78] = 0;
    res[79] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 80> quasilinear_matrix_y(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[6], 2);
    T t1 = Q[2]*t0;
    T t2 = (1.0/3.0)*t0;
    T t3 = Q[5]*t2;
    T t4 = Qaux[6]*Q[4];
    T t5 = Qaux[6]*Q[5];
    T t6 = (1.0/3.0)*t5;
    T t7 = Qaux[6]*Q[2];
    T t8 = Qaux[6]*Q[3];
    T t9 = Q[5]*t1;
    T t10 = Q[4]*t0;
    T t11 = std::pow(Q[5], 2);
    T t12 = Q[5]*t10;
    T t13 = Qaux[6]*Q[6];
    T t14 = Qaux[6]*Q[7];
    T t15 = Q[5]*t0;
    T t16 = (2.0/5.0)*Qaux[0];
    T t17 = (2.0/5.0)*Qaux[1];
    T t18 = (2.0/5.0)*t5;
    T t19 = Qaux[0]*t18;
    SimpleArray<T, 80> res;
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
    res[14] = 1;
    res[15] = 0;
    res[16] = 0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = -Q[3]*t3 - Q[4]*t1;
    res[22] = t4;
    res[23] = t6;
    res[24] = t7;
    res[25] = (1.0/3.0)*t8;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = -Q[3]*t10 - t9;
    res[32] = t5;
    res[33] = t4;
    res[34] = t8;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = p[0]*Q[1];
    res[41] = p[0]*Q[1] - std::pow(Q[4], 2)*t0 - t11*t2;
    res[42] = 0;
    res[43] = 0;
    res[44] = 2*t4;
    res[45] = (2.0/3.0)*t5;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = 0;
    res[50] = 0;
    res[51] = -2*t12;
    res[52] = 0;
    res[53] = 0;
    res[54] = 2*t5;
    res[55] = t4;
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = -Q[6]*t10 - Q[7]*t3;
    res[62] = 0;
    res[63] = 0;
    res[64] = t13;
    res[65] = (1.0/3.0)*t14;
    res[66] = t4;
    res[67] = t6;
    res[68] = 0;
    res[69] = 0;
    res[70] = 0;
    res[71] = (2.0/5.0)*Qaux[0]*Q[3]*Q[5]*t0 + (2.0/5.0)*Qaux[1]*t0*t11 - 3.0/5.0*Q[6]*t15 - Q[7]*t10 - 2.0/5.0*Q[7]*t15 - t12*t17 - t16*t9;
    res[72] = t19;
    res[73] = -t19;
    res[74] = Qaux[1]*t18 + t14;
    res[75] = -4.0/5.0*Qaux[1]*t5 - 3.0/5.0*t13 + (3.0/5.0)*t14 + t16*t7 - t16*t8 + t17*t4;
    res[76] = (3.0/5.0)*t5;
    res[77] = t18 + t4;
    res[78] = 0;
    res[79] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 8> eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 8> source(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[6], 2)*p[2];
    T t1 = Q[3]*t0;
    T t2 = Q[5]*t0;
    T t3 = (1.0/5.0)*Qaux[6];
    T t4 = Qaux[3]*t3;
    T t5 = Qaux[0]*Q[2];
    T t6 = Qaux[2]*t3;
    T t7 = Qaux[1]*Q[4];
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = -2*t1;
    res[3] = -6*t1;
    res[4] = -2*t2;
    res[5] = -6*t2;
    res[6] = 0;
    res[7] = (1.0/5.0)*Qaux[0]*Qaux[6]*Q[3]*Qaux[3] + (1.0/5.0)*Qaux[0]*Qaux[6]*Q[3]*Qaux[2] + (1.0/5.0)*Qaux[1]*Qaux[6]*Qaux[3]*Q[5] + (1.0/5.0)*Qaux[1]*Qaux[6]*Q[5]*Qaux[2] - t4*t5 - t4*t7 - t5*t6 - t6*t7;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> flux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Q[3], 2);
    T t1 = (1.0/3.0)*Qaux[6];
    T t2 = Qaux[6]*Q[2];
    T t3 = Q[3]*t1;
    T t4 = Q[4]*t2 + Q[5]*t3;
    T t5 = Q[3]*t2;
    T t6 = Q[5]*t2;
    T t7 = Qaux[6]*Q[4];
    T t8 = Q[3]*t7;
    T t9 = t6 + t8;
    T t10 = std::pow(Q[5], 2);
    T t11 = Q[5]*t7;
    T t12 = Q[5]*Q[7];
    T t13 = (3.0/5.0)*Qaux[6]*Q[6];
    T t14 = (2.0/5.0)*Qaux[6];
    T t15 = Q[3]*t14;
    T t16 = (2.0/5.0)*Qaux[0];
    T t17 = (2.0/5.0)*Qaux[1];
    T t18 = Qaux[0]*t14;
    SimpleArray<T, 16> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = Q[2];
    res[3] = Q[4];
    res[4] = Qaux[6]*std::pow(Q[2], 2) + t0*t1;
    res[5] = t4;
    res[6] = 2*t5;
    res[7] = t9;
    res[8] = t4;
    res[9] = Qaux[6]*std::pow(Q[4], 2) + t1*t10;
    res[10] = t9;
    res[11] = 2*t11;
    res[12] = Q[6]*t2 + Q[7]*t3;
    res[13] = Q[6]*t7 + t1*t12;
    res[14] = -Qaux[1]*Q[5]*t15 + Q[3]*t13 + Q[7]*t15 + Q[7]*t2 - t0*t18 + t16*t5 + t17*t8;
    res[15] = -Qaux[1]*t10*t14 - Q[3]*Q[5]*t18 + Q[5]*t13 + Q[7]*t7 + t11*t17 + t12*t14 + t16*t6;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 160> nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = p[0]*Q[1];
    T t1 = -Qaux[6]*Q[2];
    T t2 = -Qaux[6]*Q[4];
    T t3 = -6.0/5.0*Qaux[6]*Q[6] + (1.0/5.0)*Qaux[6]*Q[7];
    SimpleArray<T, 160> res;
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
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = t0;
    res[41] = 0;
    res[42] = 0;
    res[43] = 0;
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = 0;
    res[50] = 0;
    res[51] = 0;
    res[52] = 0;
    res[53] = 0;
    res[54] = 0;
    res[55] = 0;
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = 0;
    res[62] = 0;
    res[63] = 0;
    res[64] = 0;
    res[65] = 0;
    res[66] = t1;
    res[67] = 0;
    res[68] = 0;
    res[69] = 0;
    res[70] = 0;
    res[71] = t1;
    res[72] = 0;
    res[73] = 0;
    res[74] = 0;
    res[75] = 0;
    res[76] = 0;
    res[77] = 0;
    res[78] = 0;
    res[79] = 0;
    res[80] = 0;
    res[81] = t0;
    res[82] = 0;
    res[83] = 0;
    res[84] = 0;
    res[85] = 0;
    res[86] = 0;
    res[87] = 0;
    res[88] = 0;
    res[89] = 0;
    res[90] = 0;
    res[91] = 0;
    res[92] = 0;
    res[93] = 0;
    res[94] = 0;
    res[95] = 0;
    res[96] = 0;
    res[97] = 0;
    res[98] = 0;
    res[99] = 0;
    res[100] = 0;
    res[101] = 0;
    res[102] = 0;
    res[103] = 0;
    res[104] = 0;
    res[105] = 0;
    res[106] = t2;
    res[107] = 0;
    res[108] = 0;
    res[109] = 0;
    res[110] = 0;
    res[111] = t2;
    res[112] = 0;
    res[113] = 0;
    res[114] = 0;
    res[115] = 0;
    res[116] = 0;
    res[117] = 0;
    res[118] = 0;
    res[119] = 0;
    res[120] = 0;
    res[121] = 0;
    res[122] = 0;
    res[123] = 0;
    res[124] = 0;
    res[125] = 0;
    res[126] = 0;
    res[127] = 0;
    res[128] = 0;
    res[129] = 0;
    res[130] = 0;
    res[131] = 0;
    res[132] = 0;
    res[133] = 0;
    res[134] = 0;
    res[135] = 0;
    res[136] = 0;
    res[137] = 0;
    res[138] = 0;
    res[139] = 0;
    res[140] = 0;
    res[141] = 0;
    res[142] = 0;
    res[143] = 0;
    res[144] = 0;
    res[145] = 0;
    res[146] = t3;
    res[147] = 0;
    res[148] = 0;
    res[149] = 0;
    res[150] = 0;
    res[151] = t3;
    res[152] = 0;
    res[153] = 0;
    res[154] = 0;
    res[155] = 0;
    res[156] = 0;
    res[157] = 0;
    res[158] = 0;
    res[159] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 80> source_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[6], 3)*p[2];
    T t1 = Q[3]*t0;
    T t2 = std::pow(Qaux[6], 2);
    T t3 = p[2]*t2;
    T t4 = -2*t3;
    T t5 = -6*t3;
    T t6 = Q[5]*t0;
    T t7 = (1.0/5.0)*t2;
    T t8 = Qaux[3]*t7;
    T t9 = Qaux[0]*Q[2];
    T t10 = Qaux[2]*t7;
    T t11 = Qaux[0]*Q[3];
    T t12 = Qaux[1]*Q[4];
    T t13 = Qaux[1]*Q[5];
    T t14 = (1.0/5.0)*Qaux[6];
    T t15 = Qaux[0]*t14;
    T t16 = Qaux[3]*t15 + Qaux[2]*t15;
    T t17 = Qaux[1]*t14;
    T t18 = Qaux[3]*t17 + Qaux[2]*t17;
    SimpleArray<T, 80> res;
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
    res[20] = 0;
    res[21] = 4*t1;
    res[22] = 0;
    res[23] = t4;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 12*t1;
    res[32] = 0;
    res[33] = t5;
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = 4*t6;
    res[42] = 0;
    res[43] = 0;
    res[44] = 0;
    res[45] = t4;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = 0;
    res[50] = 0;
    res[51] = 12*t6;
    res[52] = 0;
    res[53] = 0;
    res[54] = 0;
    res[55] = t5;
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = 0;
    res[62] = 0;
    res[63] = 0;
    res[64] = 0;
    res[65] = 0;
    res[66] = 0;
    res[67] = 0;
    res[68] = 0;
    res[69] = 0;
    res[70] = 0;
    res[71] = -t10*t11 + t10*t12 - t10*t13 + t10*t9 - t11*t8 + t12*t8 - t13*t8 + t8*t9;
    res[72] = -t16;
    res[73] = t16;
    res[74] = -t18;
    res[75] = t18;
    res[76] = 0;
    res[77] = 0;
    res[78] = 0;
    res[79] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 56> source_jacobian_wrt_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = (1.0/5.0)*Qaux[6];
    T t1 = Q[2]*t0;
    T t2 = Q[4]*t0;
    T t3 = (1.0/5.0)*Qaux[0]*Qaux[6]*Q[3] - Qaux[0]*t1 + (1.0/5.0)*Qaux[1]*Qaux[6]*Q[5] - Qaux[1]*t2;
    SimpleArray<T, 56> res;
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
    res[43] = 0;
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = (1.0/5.0)*Qaux[6]*Q[3]*Qaux[3] + (1.0/5.0)*Qaux[6]*Q[3]*Qaux[2] - Qaux[3]*t1 - Qaux[2]*t1;
    res[50] = (1.0/5.0)*Qaux[6]*Qaux[3]*Q[5] + (1.0/5.0)*Qaux[6]*Q[5]*Qaux[2] - Qaux[3]*t2 - Qaux[2]*t2;
    res[51] = t3;
    res[52] = t3;
    res[53] = 0;
    res[54] = 0;
    res[55] = 0;
    return res;
    }

    // update_variables: NOT emitted from the SystemModel (provide solver-side).
    //   no pointwise state remap on this SystemModel; the solver
    //   leaves Q unchanged after the transport step.

    PORTABLE_FN static inline SimpleArray<T, 7> update_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T dt)
    {
    SimpleArray<T, 7> res;
    res[0] = Qaux[0];
    res[1] = Qaux[1];
    res[2] = Qaux[2];
    res[3] = Qaux[3];
    res[4] = Qaux[4];
    res[5] = Qaux[5];
    res[6] = std::pow((T)2, 1.0/2.0)*std::max((T)0, Q[1])/std::pow(std::pow(Q[1], 4) + std::pow(std::max((T)1.0e-8, Q[1]), 4), 1.0/2.0);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 70> update_aux_variables_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T dt)
    {
    T t0 = std::pow((T)2, 1.0/2.0);
    T t1 = std::max((T)1.0e-8, Q[1]);
    T t2 = std::pow(Q[1], 4) + std::pow(t1, 4);
    SimpleArray<T, 70> res;
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
    res[13] = t0*Heaviside(Q[1], 1.0/2.0)/std::pow(t2, 1.0/2.0) + t0*(-2*std::pow(Q[1], 3) - 2*std::pow(t1, 3)*Heaviside(Q[1] - 1.0e-8, 1.0/2.0))*std::max((T)0, Q[1])/std::pow(t2, 3.0/2.0);
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
    res[43] = 0;
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = 0;
    res[48] = 0;
    res[49] = 0;
    res[50] = 0;
    res[51] = 0;
    res[52] = 0;
    res[53] = 0;
    res[54] = 0;
    res[55] = 0;
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = 0;
    res[62] = 0;
    res[63] = 0;
    res[64] = 0;
    res[65] = 0;
    res[66] = 0;
    res[67] = 0;
    res[68] = 0;
    res[69] = 0;
    return res;
    }

    static inline void update_aux_variables(
        T* const* Q,
        T* const* Qaux,
        const T dt,
        const ZoomyMesh& mesh)
    {
    // Qaux[0] (b_x) = D^(1, 0) Q[0]
    compute_derivative(Qaux[0], Q[0], 1, 0, 0, mesh);
    // Qaux[1] (b_y) = D^(0, 1) Q[0]
    compute_derivative(Qaux[1], Q[0], 0, 1, 0, mesh);
    // Qaux[3] (q_x_1_x) = D^(1, 0) Q[3]
    compute_derivative(Qaux[3], Q[3], 1, 0, 0, mesh);
    // Qaux[2] (q_y_1_y) = D^(0, 1) Q[5]
    compute_derivative(Qaux[2], Q[5], 0, 1, 0, mesh);
    }

    // initial_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel's IC is trivial / zero; set it case-side.

    // initial_aux_condition: NOT emitted from the SystemModel (provide solver-side).
    //   the SystemModel's IC is trivial / zero; set it case-side.

    PORTABLE_FN static inline SimpleArray<T, 10> boundary_conditions(
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
        SimpleArray<T, 10> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = 0.0;
        res[3] = 0.0;
        res[4] = 0.11197;
        res[5] = 0.0;
        res[6] = 0.0;
        res[7] = 0.0;
        res[8] = Q[8];
        res[9] = Q[9];
        return res;
    } else if (bc_idx == 1 || bc_idx == 2) {
        SimpleArray<T, 10> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = Q[2];
        res[3] = Q[3];
        res[4] = Q[4];
        res[5] = Q[5];
        res[6] = Q[6];
        res[7] = Q[7];
        res[8] = Q[8];
        res[9] = Q[9];
        return res;
    } else if (bc_idx == 3) {
        SimpleArray<T, 10> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = Q[2];
        res[3] = Q[3];
        res[4] = Q[4];
        res[5] = Q[5];
        res[6] = Q[6];
        res[7] = Q[7];
        res[8] = 0.0;
        res[9] = 0.0;
        return res;
    }
    SimpleArray<T, 10> default_res = {0};
    return default_res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> aux_boundary_conditions(
        const int bc_idx,
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n,
        const T* X,
        const T time,
        const T dX)
    {
    if (bc_idx == 0 || bc_idx == 1 || bc_idx == 2 || bc_idx == 3) {
        SimpleArray<T, 6> res;
        res[0] = Qaux[0];
        res[1] = Qaux[1];
        res[2] = Qaux[2];
        res[3] = Qaux[3];
        res[4] = Qaux[4];
        res[5] = Qaux[5];
        return res;
    }
    SimpleArray<T, 6> default_res = {0};
    return default_res;
    }

};
