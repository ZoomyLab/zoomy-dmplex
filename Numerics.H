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
    T t5 = std::pow((T)2, 1.0/2.0);
    T t6 = std::max((T)0, t1);
    T t7 = std::pow(t2, 4) + std::pow(std::max(t1, p[5]), 4);
    T t8 = (1.0 / std::pow(std::max((T)1.0e-14, Q_minus[1]), 1));
    T t9 = t2*t8;
    T t10 = Q_minus[2]*t9;
    T t11 = Q_minus[3]*t9;
    T t12 = std::max((T)0, t3);
    T t13 = std::pow(t4, 4) + std::pow(std::max(t3, p[5]), 4);
    T t14 = (1.0 / std::pow(std::max((T)1.0e-14, Q_plus[1]), 1));
    T t15 = t14*t4;
    T t16 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t17 = t16*std::pow(t2, 5.0/2.0);
    T t18 = std::pow(t2, 2);
    T t19 = t18*t8;
    T t20 = Q_minus[2]*n[0]*t19 + Q_minus[3]*n[1]*t19;
    T t21 = 2*std::pow(t6, 2)/t7;
    T t22 = t16*std::pow(t4, 5.0/2.0);
    T t23 = std::pow(t4, 2);
    T t24 = t14*t23;
    T t25 = Q_plus[2]*n[0]*t24 + Q_plus[3]*n[1]*t24;
    T t26 = 2*std::pow(t12, 2)/t13;
    // LOCAL FIX (TODO -> core codegen): the generated desingularized advective wave
    // speed |q.n|(h*)^2/h / sqrt((h*)^4 + max(h*,eps)^4) collapses by (h*/eps)^2 when
    // h* < eps, starving the Rusanov dissipation at steep wet/dry fronts -> h<0. Use
    // the physical speed lambda = max over both states of |u.n| + sqrt(g h*)|n|, with
    // u = q/max(1e-14,h) the TRUE velocity (cell h), h* the Audusse depth (t2,t4).
    T nnorm = std::sqrt(n[0]*n[0] + n[1]*n[1]);
    T uLn = (n[0]*Q_minus[2] + n[1]*Q_minus[3]) / std::max((T)1.0e-14, Q_minus[1]);
    T uRn = (n[0]*Q_plus[2]  + n[1]*Q_plus[3])  / std::max((T)1.0e-14, Q_plus[1]);
    T t27 = 0.5*std::max(std::fabs(uLn) + std::sqrt(p[0]*t2)*nnorm,
                         std::fabs(uRn) + std::sqrt(p[0]*t4)*nnorm);
    T t28 = t27*(-t2 + t4);
    T t29 = (1.0/2.0)*p[0];
    T t30 = -std::pow(Q_plus[1], 2)*t29 + (1.0/2.0)*p[0]*t23;
    T t31 = t27*(Q_plus[2]*t14*t4 - t10);
    T t32 = t27*(Q_plus[3]*t14*t4 - t11);
    T t33 = std::pow(Q_minus[1], 2)*t29 - t18*t29;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = t28;
    res[2] = n[0]*t30 + t31;
    res[3] = n[1]*t30 + t32;
    res[4] = 0;
    res[5] = -t28;
    res[6] = n[0]*t33 - t31;
    res[7] = n[1]*t33 - t32;
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
    T t2 = std::pow(Qaux[6], 2);
    T t3 = std::pow(p[0], 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0)*std::pow(std::max((T)0, std::pow(Q[1], 5)), 1.0/2.0);
    T t4 = Q[1]*t0 + Q[1]*t1;
    SimpleArray<T, 1> res;
    res[0] = std::max(Qaux[6]*std::fabs(t0 + t1), std::max(t2*std::fabs(t3 + t4), t2*std::fabs(t3 - t4)));
    return res;
    }

};
