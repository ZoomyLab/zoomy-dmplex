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
    T t0 = std::max((T)1.0e-14, Q_minus[1]);
    T t1 = -std::max(Q_minus[0], Q_plus[0]);
    T t2 = Q_minus[0] + Q_minus[1] + t1;
    T t3 = std::max((T)0.0, t2);
    T t4 = t3/t0;
    T t5 = 0.5*n[0];
    T t6 = Q_minus[2]*t5;
    T t7 = 0.5*n[1];
    T t8 = Q_minus[3]*t7;
    T t9 = std::max((T)1.0e-14, Q_plus[1]);
    T t10 = Q_plus[0] + Q_plus[1] + t1;
    T t11 = std::max((T)0.0, t10);
    T t12 = t11/t9;
    T t13 = Q_plus[2]*t5;
    T t14 = Q_plus[3]*t7;
    T t15 = std::pow(t3, 2);
    T t16 = std::pow((T)2, 1.0/2.0);
    T t17 = t15*t16*std::max((T)0, t2)/(std::pow(t0, 2)*std::pow(std::pow(t3, 4) + std::pow(std::max(t2, p[5]), 4), 1.0/2.0));
    T t18 = std::pow(t11, 2);
    T t19 = t16*t18*std::max((T)0, t10)/(std::pow(t9, 2)*std::pow(std::pow(t11, 4) + std::pow(std::max(t10, p[5]), 4), 1.0/2.0));
    T t20 = (1.0/2.0)*p[0];
    T t21 = t15*t20;
    T t22 = t18*t20;
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = t12*t13 + t12*t14 + t4*t6 + t4*t8;
    res[2] = Q_minus[2]*t17*t8 + Q_plus[2]*t14*t19 + t5*(std::pow(Q_minus[2], 2)*t17 + t21) + t5*(std::pow(Q_plus[2], 2)*t19 + t22);
    res[3] = Q_minus[3]*t17*t6 + Q_plus[3]*t13*t19 + t7*(std::pow(Q_minus[3], 2)*t17 + t21) + t7*(std::pow(Q_plus[3], 2)*t19 + t22);
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
    T t1 = Q_minus[0] + Q_minus[1] + t0;
    T t2 = std::max((T)0.0, t1);
    T t3 = Q_plus[0] + Q_plus[1] + t0;
    T t4 = std::max((T)0.0, t3);
    T t5 = std::max((T)1.0e-14, t1);
    T t6 = (1.0 / std::pow(std::max((T)1.0e-14, Q_minus[1]), 1));
    T t7 = t2*t6;
    T t8 = Q_minus[2]*t7;
    T t9 = Q_minus[3]*t7;
    T t10 = std::max((T)1.0e-14, t3);
    T t11 = (1.0 / std::pow(std::max((T)1.0e-14, Q_plus[1]), 1));
    T t12 = t11*t4;
    T t13 = (1.0 / std::pow(t5, 2));
    T t14 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t15 = t14*std::pow(t2, 5.0/2.0);
    T t16 = std::pow(t2, 2);
    T t17 = t16*t6;
    T t18 = Q_minus[2]*n[0]*t17 + Q_minus[3]*n[1]*t17;
    T t19 = (1.0 / std::pow(t10, 2));
    T t20 = t14*std::pow(t4, 5.0/2.0);
    T t21 = std::pow(t4, 2);
    T t22 = t11*t21;
    T t23 = Q_plus[2]*n[0]*t22 + Q_plus[3]*n[1]*t22;
    T t24 = 0.5*std::max(t13*std::fabs(t15 + t18), std::max(t19*std::fabs(t20 + t23), std::max(t13*std::fabs(t15 - t18), std::max(t19*std::fabs(t20 - t23), std::max(std::fabs(n[0]*t8 + n[1]*t9)/t5, std::fabs(Q_plus[2]*n[0]*t12 + Q_plus[3]*n[1]*t12)/t10)))));
    T t25 = t24*(-t2 + t4);
    T t26 = (1.0/2.0)*p[0];
    T t27 = -std::pow(Q_plus[1], 2)*t26 + (1.0/2.0)*p[0]*t21;
    T t28 = t24*(Q_plus[2]*t11*t4 - t8);
    T t29 = t24*(Q_plus[3]*t11*t4 - t9);
    T t30 = std::pow(Q_minus[1], 2)*t26 - t16*t26;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = t25;
    res[2] = n[0]*t27 + t28;
    res[3] = n[1]*t27 + t29;
    res[4] = 0;
    res[5] = -t25;
    res[6] = n[0]*t30 - t28;
    res[7] = n[1]*t30 - t29;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = std::max((T)1.0e-14, Q[1]);
    T t1 = n[0]*Q[2];
    T t2 = n[1]*Q[3];
    T t3 = (1.0 / std::pow(t0, 2));
    T t4 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0)*std::pow(std::max((T)0, std::pow(Q[1], 5)), 1.0/2.0);
    T t5 = Q[1]*t1 + Q[1]*t2;
    SimpleArray<T, 1> res;
    res[0] = std::max(t3*std::fabs(t4 + t5), std::max(std::fabs(t1 + t2)/t0, t3*std::fabs(t4 - t5)));
    return res;
    }

};
