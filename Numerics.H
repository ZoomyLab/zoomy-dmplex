#pragma once
#include <cmath>
#include <array>
#include "Model.H"

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
#include <algorithm>

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif

template <typename T>
struct Numerics {
    static constexpr int n_dof_q = 4;
    PORTABLE_FN static inline SimpleArray<T, 4> numerical_flux(
        const T* Q_minus,
        const T* Q_plus,
        const T* Qaux_minus,
        const T* Qaux_plus,
        const T* p,
        const T* n)
    {
    T t0 = -std::max(Q_minus[0], Q_plus[0]);
    T t1 = std::max(0.0, Q_minus[0] + Q_minus[1] + t0);
    T t2 = std::max(1.0e-14, Q_minus[1]);
    T t3 = t1/t2;
    T t4 = 0.5*n[0];
    T t5 = Q_minus[2]*t4;
    T t6 = 0.5*n[1];
    T t7 = Q_minus[3]*t6;
    T t8 = std::max(0.0, Q_plus[0] + Q_plus[1] + t0);
    T t9 = std::max(1.0e-14, Q_plus[1]);
    T t10 = t8/t9;
    T t11 = Q_plus[2]*t4;
    T t12 = Q_plus[3]*t6;
    T t13 = t1/std::pow(t2, 2);
    T t14 = t8/std::pow(t9, 2);
    T t15 = (1.0/2.0)*p[0];
    T t16 = std::pow(t1, 2)*t15;
    T t17 = t15*std::pow(t8, 2);
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = t10*t11 + t10*t12 + t3*t5 + t3*t7;
    res[2] = Q_minus[2]*t13*t7 + Q_plus[2]*t12*t14 + t4*(std::pow(Q_minus[2], 2)*t13 + t16) + t4*(std::pow(Q_plus[2], 2)*t14 + t17);
    res[3] = Q_minus[3]*t13*t5 + Q_plus[3]*t11*t14 + t6*(std::pow(Q_minus[3], 2)*t13 + t16) + t6*(std::pow(Q_plus[3], 2)*t14 + t17);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 8> numerical_fluctuations(
        const T* Q_minus,
        const T* Q_plus,
        const T* Qaux_minus,
        const T* Qaux_plus,
        const T* p,
        const T* n)
    {
    T t0 = -std::max(Q_minus[0], Q_plus[0]);
    T t1 = std::max(0.0, Q_minus[0] + Q_minus[1] + t0);
    T t2 = std::max(0.0, Q_plus[0] + Q_plus[1] + t0);
    T t3 = (1.0 / std::pow(std::max(1.0e-14, Q_minus[1]), 1));
    T t4 = t1*t3;
    T t5 = Q_minus[2]*t4;
    T t6 = Q_minus[3]*t4;
    T t7 = (1.0 / std::pow(std::max(1.0e-14, Q_plus[1]), 1));
    T t8 = t2*t7;
    T t9 = std::pow(t1, 2);
    T t10 = (1.0 / std::pow(t9, 1));
    T t11 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t12 = std::pow(t1, 5.0/2.0)*t11;
    T t13 = t3*t9;
    T t14 = Q_minus[2]*n[0]*t13 + Q_minus[3]*n[1]*t13;
    T t15 = std::pow(t2, 2);
    T t16 = (1.0 / std::pow(t15, 1));
    T t17 = t11*std::pow(t2, 5.0/2.0);
    T t18 = t15*t7;
    T t19 = Q_plus[2]*n[0]*t18 + Q_plus[3]*n[1]*t18;
    T t20 = 0.5*std::max(t10*std::fabs(t12 + t14), std::max(t16*std::fabs(t17 + t19), std::max(t10*std::fabs(t12 - t14), std::max(t16*std::fabs(t17 - t19), std::max(std::fabs(n[0]*t5 + n[1]*t6)/t1, std::fabs(Q_plus[2]*n[0]*t8 + Q_plus[3]*n[1]*t8)/t2)))));
    T t21 = t20*(-t1 + t2);
    T t22 = (1.0/2.0)*p[0];
    T t23 = -std::pow(Q_plus[1], 2)*t22 + (1.0/2.0)*p[0]*t15;
    T t24 = t20*(Q_plus[2]*t2*t7 - t5);
    T t25 = t20*(Q_plus[3]*t2*t7 - t6);
    T t26 = std::pow(Q_minus[1], 2)*t22 - t22*t9;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = t21;
    res[2] = n[0]*t23 + t24;
    res[3] = n[1]*t23 + t25;
    res[4] = 0;
    res[5] = -t21;
    res[6] = n[0]*t26 - t24;
    res[7] = n[1]*t26 - t25;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
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
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(t2*(t3 + t4)), std::max(std::fabs((t0 + t1)/Q[1]), std::fabs(t2*(t3 - t4))));
    return res;
    }

};
