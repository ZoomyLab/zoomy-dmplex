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
    T t2 = std::max((T)0.0, Q_minus[0] + Q_minus[1] + t1);
    T t3 = t2/t0;
    T t4 = 0.5*n[0];
    T t5 = Q_minus[2]*t4;
    T t6 = 0.5*n[1];
    T t7 = Q_minus[3]*t6;
    T t8 = std::max((T)1.0e-14, Q_plus[1]);
    T t9 = std::max((T)0.0, Q_plus[0] + Q_plus[1] + t1);
    T t10 = t9/t8;
    T t11 = Q_plus[2]*t4;
    T t12 = Q_plus[3]*t6;
    T t13 = std::pow(t2, 2);
    T t14 = t13/(std::pow(t0, 2)*(t2 + 1.0e-14));
    T t15 = std::pow(t9, 2);
    T t16 = t15/(std::pow(t8, 2)*(t9 + 1.0e-14));
    T t17 = (1.0/2.0)*p[0];
    T t18 = t13*t17;
    T t19 = t15*t17;
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = t10*t11 + t10*t12 + t3*t5 + t3*t7;
    res[2] = Q_minus[2]*t14*t7 + Q_plus[2]*t12*t16 + t4*(std::pow(Q_minus[2], 2)*t14 + t18) + t4*(std::pow(Q_plus[2], 2)*t16 + t19);
    res[3] = Q_minus[3]*t14*t5 + Q_plus[3]*t11*t16 + t6*(std::pow(Q_minus[3], 2)*t14 + t18) + t6*(std::pow(Q_plus[3], 2)*t16 + t19);
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
    T t5 = (1.0 / std::pow(t2 + 1.0e-14, 1));
    T t6 = t2/std::max((T)1.0e-14, Q_minus[1]);
    T t7 = Q_minus[2]*t6;
    T t8 = Q_minus[3]*t6;
    T t9 = n[0]*t7 + n[1]*t8;
    T t10 = (1.0 / std::pow(t4 + 1.0e-14, 1));
    T t11 = (1.0 / std::pow(std::max((T)1.0e-14, Q_plus[1]), 1));
    T t12 = t11*t4;
    T t13 = Q_plus[2]*n[0]*t12 + Q_plus[3]*n[1]*t12;
    T t14 = std::pow(p[0], 1.0/2.0);
    T t15 = t14*std::pow(std::max(t1, p[5]), 1.0/2.0);
    T t16 = t5*t9;
    T t17 = t14*std::pow(std::max(t3, p[5]), 1.0/2.0);
    T t18 = t10*t13;
    T t19 = 0.5*std::max(t10*std::fabs(t13), std::max(t5*std::fabs(t9), std::max(std::fabs(t15 + t16), std::max(std::fabs(t17 + t18), std::max(std::fabs(t15 - t16), std::fabs(t17 - t18))))));
    T t20 = t19*(-t2 + t4);
    T t21 = (1.0/2.0)*p[0];
    T t22 = -std::pow(Q_plus[1], 2)*t21 + (1.0/2.0)*p[0]*std::pow(t4, 2);
    T t23 = t19*(Q_plus[2]*t11*t4 - t7);
    T t24 = t19*(Q_plus[3]*t11*t4 - t8);
    T t25 = std::pow(Q_minus[1], 2)*t21 - std::pow(t2, 2)*t21;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = t20;
    res[2] = n[0]*t22 + t23;
    res[3] = n[1]*t22 + t24;
    res[4] = 0;
    res[5] = -t20;
    res[6] = n[0]*t25 - t23;
    res[7] = n[1]*t25 - t24;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = n[0]*Q[2] + n[1]*Q[3];
    T t1 = std::pow(p[0], 1.0/2.0)*std::pow(std::max(Q[1], p[5]), 1.0/2.0);
    T t2 = Qaux[6]*t0;
    SimpleArray<T, 1> res;
    res[0] = std::max(Qaux[6]*std::fabs(t0), std::max(std::fabs(t1 + t2), std::fabs(t1 - t2)));
    return res;
    }

};
