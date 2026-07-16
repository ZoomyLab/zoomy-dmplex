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
    static constexpr int n_dof_q = 3;
    PORTABLE_FN static inline SimpleArray<T, 3> numerical_flux(
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
    T t4 = 0.5*n[0];
    T t5 = std::max((T)1.0e-14, Q_plus[1]);
    T t6 = Q_plus[0] + Q_plus[1] + t1;
    T t7 = std::max((T)0.0, t6);
    T t8 = std::pow(t3, 2);
    T t9 = (1.0/2.0)*p[0];
    T t10 = std::pow((T)2, 1.0/2.0);
    T t11 = std::pow(t7, 2);
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = Q_minus[2]*t3*t4/t0 + Q_plus[2]*t4*t7/t5;
    res[2] = t4*(std::pow(Q_minus[2], 2)*t10*t8*std::max((T)0, t2)/(std::pow(t0, 2)*std::pow(std::pow(t3, 4) + std::pow(std::max((T)9.9999999999999998e-13, t2), 4), 1.0/2.0)) + t8*t9) + t4*(std::pow(Q_plus[2], 2)*t10*t11*std::max((T)0, t6)/(std::pow(t5, 2)*std::pow(std::pow(t7, 4) + std::pow(std::max((T)9.9999999999999998e-13, t6), 4), 1.0/2.0)) + t11*t9);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> numerical_fluctuations(
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
    T t6 = std::pow(p[0], 1.0/2.0)*n[0];
    T t7 = std::pow(t2, 5.0/2.0)*t6/std::pow(t5, 2);
    T t8 = Q_minus[2]*t2/std::max((T)1.0e-14, Q_minus[1]);
    T t9 = n[0]*t8/t5;
    T t10 = std::max((T)1.0e-14, t3);
    T t11 = std::pow(t4, 5.0/2.0)*t6/std::pow(t10, 2);
    T t12 = (1.0 / std::pow(std::max((T)1.0e-14, Q_plus[1]), 1));
    T t13 = Q_plus[2]*n[0]*t12*t4/t10;
    T t14 = 0.5*std::max(std::fabs(t11 + t13), std::max(std::fabs(t7 + t9), std::max(std::fabs(t11 - t13), std::fabs(t7 - t9))));
    T t15 = t14*(-t2 + t4);
    T t16 = (1.0/2.0)*p[0];
    T t17 = t14*(Q_plus[2]*t12*t4 - t8);
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = t15;
    res[2] = n[0]*(-std::pow(Q_plus[1], 2)*t16 + (1.0/2.0)*p[0]*std::pow(t4, 2)) + t17;
    res[3] = 0;
    res[4] = -t15;
    res[5] = n[0]*(std::pow(Q_minus[1], 2)*t16 - t16*std::pow(t2, 2)) - t17;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = std::max((T)1.0e-14, Q[1]);
    T t1 = n[0]*Q[2]/t0;
    T t2 = std::pow(p[0], 1.0/2.0)*n[0]*std::pow(std::max((T)0, std::pow(Q[1], 5)), 1.0/2.0)/std::pow(t0, 2);
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(t1 + t2), std::fabs(t1 - t2));
    return res;
    }

};
