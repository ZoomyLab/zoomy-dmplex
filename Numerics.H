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
    T t17 = t15*t16*std::max((T)0, t2)/(std::pow(t0, 2)*std::pow(std::pow(t3, 4) + std::pow(std::max((T)9.9999999999999998e-13, t2), 4), 1.0/2.0));
    T t18 = std::pow(t11, 2);
    T t19 = t16*t18*std::max((T)0, t10)/(std::pow(t9, 2)*std::pow(std::pow(t11, 4) + std::pow(std::max((T)9.9999999999999998e-13, t10), 4), 1.0/2.0));
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
    T t5 = t2 > 1.0e-8;
    T t6 = t4 > 1.0e-8;
    T t7 = std::pow((T)2, 1.0/2.0);
    T t8 = std::max((T)0, t1);
    T t9 = std::pow(t2, 4) + std::pow(std::max((T)9.9999999999999998e-13, t1), 4);
    T t10 = (1.0 / std::pow(std::max((T)1.0e-14, Q_minus[1]), 1));
    T t11 = t10*t2;
    T t12 = Q_minus[2]*t11;
    T t13 = Q_minus[3]*t11;
    T t14 = std::max((T)0, t3);
    T t15 = std::pow(t4, 4) + std::pow(std::max((T)9.9999999999999998e-13, t3), 4);
    T t16 = (1.0 / std::pow(std::max((T)1.0e-14, Q_plus[1]), 1));
    T t17 = t16*t4;
    T t18 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t19 = t18*std::pow(t2, 5.0/2.0);
    T t20 = std::pow(t2, 2);
    T t21 = t10*t20;
    T t22 = Q_minus[2]*n[0]*t21 + Q_minus[3]*n[1]*t21;
    T t23 = 2*std::pow(t8, 2)/t9;
    T t24 = t18*std::pow(t4, 5.0/2.0);
    T t25 = std::pow(t4, 2);
    T t26 = t16*t25;
    T t27 = Q_plus[2]*n[0]*t26 + Q_plus[3]*n[1]*t26;
    T t28 = 2*std::pow(t14, 2)/t15;
    T t29 = 0.5*std::max(std::fabs(((t5) ? (0) : (0))), std::max(std::fabs(((t6) ? (0) : (0))), std::max(std::fabs(((t5) ? (t23*(t19 + t22)) : (0))), std::max(std::fabs(((t6) ? (t28*(t24 + t27)) : (0))), std::max(std::fabs(((t5) ? (t23*(-t19 + t22)) : (0))), std::max(std::fabs(((t6) ? (t28*(-t24 + t27)) : (0))), std::max(std::fabs(((t5) ? (t7*t8*(n[0]*t12 + n[1]*t13)/std::pow(t9, 1.0/2.0)) : (0))), std::fabs(((t6) ? (t14*t7*(Q_plus[2]*n[0]*t17 + Q_plus[3]*n[1]*t17)/std::pow(t15, 1.0/2.0)) : (0))))))))));
    T t30 = t29*(-t2 + t4);
    T t31 = (1.0/2.0)*p[0];
    T t32 = -std::pow(Q_plus[1], 2)*t31 + (1.0/2.0)*p[0]*t25;
    T t33 = t29*(Q_plus[2]*t16*t4 - t12);
    T t34 = t29*(Q_plus[3]*t16*t4 - t13);
    T t35 = std::pow(Q_minus[1], 2)*t31 - t20*t31;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = t30;
    res[2] = n[0]*t32 + t33;
    res[3] = n[1]*t32 + t34;
    res[4] = 0;
    res[5] = -t30;
    res[6] = n[0]*t35 - t33;
    res[7] = n[1]*t35 - t34;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = Q[1] > 1.0e-8;
    T t1 = n[0]*Q[2];
    T t2 = n[1]*Q[3];
    T t3 = std::pow(Qaux[8], 2);
    T t4 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0)*std::pow(std::pow(Q[1], 5), 1.0/2.0);
    T t5 = Q[1]*t1 + Q[1]*t2;
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(((t0) ? (0) : (0))), std::max(std::fabs(((t0) ? (Qaux[8]*(t1 + t2)) : (0))), std::max(std::fabs(((t0) ? (t3*(t4 + t5)) : (0))), std::fabs(((t0) ? (t3*(-t4 + t5)) : (0))))));
    return res;
    }

};
