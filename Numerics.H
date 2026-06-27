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
    T t0 = -std::max(Q_minus[0], Q_plus[0]);
    T t1 = std::max(0.0, Q_minus[0] + Q_minus[1] + t0);
    T t2 = std::max(1.0e-14, Q_minus[1]);
    T t3 = 0.5*n[0];
    T t4 = std::max(0.0, Q_plus[0] + Q_plus[1] + t0);
    T t5 = std::max(1.0e-14, Q_plus[1]);
    T t6 = (1.0/2.0)*p[0];
    SimpleArray<T, 3> res;
    res[0] = 0;
    res[1] = Q_minus[2]*t1*t3/t2 + Q_plus[2]*t3*t4/t5;
    res[2] = t3*(std::pow(Q_minus[2], 2)*t1/std::pow(t2, 2) + std::pow(t1, 2)*t6) + t3*(std::pow(Q_plus[2], 2)*t4/std::pow(t5, 2) + std::pow(t4, 2)*t6);
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
    T t1 = std::max(0.0, Q_minus[0] + Q_minus[1] + t0);
    T t2 = std::max(0.0, Q_plus[0] + Q_plus[1] + t0);
    T t3 = Q_minus[2]/std::max(1.0e-14, Q_minus[1]);
    T t4 = n[0]*t3;
    T t5 = std::pow(p[0], 1.0/2.0)*n[0];
    T t6 = std::pow(t1, 1.0/2.0)*t5;
    T t7 = (1.0 / std::pow(std::max(1.0e-14, Q_plus[1]), 1));
    T t8 = Q_plus[2]*n[0]*t7;
    T t9 = std::pow(t2, 1.0/2.0)*t5;
    T t10 = 0.5*std::max(std::fabs(t4 + t6), std::max(std::fabs(t8 + t9), std::max(std::fabs(t4 - t6), std::fabs(t8 - t9))));
    T t11 = t10*(-t1 + t2);
    T t12 = (1.0/2.0)*p[0];
    T t13 = t10*(Q_plus[2]*t2*t7 - t1*t3);
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = t11;
    res[2] = n[0]*(-std::pow(Q_plus[1], 2)*t12 + (1.0/2.0)*p[0]*std::pow(t2, 2)) + t13;
    res[3] = 0;
    res[4] = -t11;
    res[5] = n[0]*(std::pow(Q_minus[1], 2)*t12 - std::pow(t1, 2)*t12) - t13;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = n[0]*Q[2]/Q[1];
    T t1 = std::pow(p[0], 1.0/2.0)*n[0]*std::pow(std::pow(Q[1], 5), 1.0/2.0)/std::pow(Q[1], 2);
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(t0 + t1), std::fabs(t0 - t1));
    return res;
    }

};
