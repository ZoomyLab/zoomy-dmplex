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
    static constexpr int n_dof_q = 6;
    PORTABLE_FN static inline SimpleArray<T, 6> numerical_flux(
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
    T t11 = t10*t8*std::max((T)0, t2)/(std::pow(t0, 2)*std::pow(std::pow(t3, 4) + std::pow(std::max((T)9.9999999999999998e-13, t2), 4), 1.0/2.0));
    T t12 = std::pow(Q_minus[3], 2);
    T t13 = (1.0/3.0)*t11;
    T t14 = std::pow(t7, 2);
    T t15 = t10*t14*std::max((T)0, t6)/(std::pow(t5, 2)*std::pow(std::pow(t7, 4) + std::pow(std::max((T)9.9999999999999998e-13, t6), 4), 1.0/2.0));
    T t16 = std::pow(Q_plus[3], 2);
    T t17 = (1.0/3.0)*t15;
    T t18 = 1.0*n[0];
    T t19 = Q_minus[2]*t11;
    T t20 = Q_minus[3]*t19;
    T t21 = Q_plus[2]*t15;
    T t22 = Q_plus[3]*t21;
    T t23 = Q_minus[3]*Q_minus[5];
    T t24 = Q_plus[3]*Q_plus[5];
    T t25 = (2.0/5.0)*t11;
    T t26 = (2.0/5.0)*t15;
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = Q_minus[2]*t3*t4/t0 + Q_plus[2]*t4*t7/t5;
    res[2] = t4*(std::pow(Q_minus[2], 2)*t11 + t12*t13 + t8*t9) + t4*(std::pow(Q_plus[2], 2)*t15 + t14*t9 + t16*t17);
    res[3] = t18*t20 + t18*t22;
    res[4] = t4*(Q_minus[4]*t19 + t13*t23) + t4*(Q_plus[4]*t21 + t17*t24);
    res[5] = t4*((3.0/5.0)*Q_minus[3]*Q_minus[4]*t11 + Q_minus[5]*t19 - Qaux_minus[0]*t12*t25 + (2.0/5.0)*Qaux_minus[0]*t20 + t23*t25) + t4*((3.0/5.0)*Q_plus[3]*Q_plus[4]*t15 + Q_plus[5]*t21 - Qaux_plus[0]*t16*t26 + (2.0/5.0)*Qaux_plus[0]*t22 + t24*t26);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 12> numerical_fluctuations(
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
    T t5 = std::fabs(n[0]);
    T t6 = Q_minus[2]*n[0];
    T t7 = std::max((T)1.0e-14, Q_minus[1]);
    T t8 = t2/t7;
    T t9 = std::pow((T)2, 1.0/2.0);
    T t10 = std::pow(t2, 4) + std::pow(std::max((T)9.9999999999999998e-13, t1), 4);
    T t11 = std::max((T)0, t1);
    T t12 = t11/std::pow(t10, 1.0/2.0);
    T t13 = t12*t9;
    T t14 = t13*t8;
    T t15 = t14*std::fabs(t6);
    T t16 = t14*std::fabs(Q_minus[3]*n[0]);
    T t17 = std::pow(t2, 2);
    T t18 = std::pow(t11, 2)*t17/(t10*std::pow(t7, 2));
    T t19 = Q_plus[2]*n[0];
    T t20 = std::max((T)1.0e-14, Q_plus[1]);
    T t21 = (1.0 / std::pow(t20, 1));
    T t22 = t21*t4;
    T t23 = std::pow(t4, 4) + std::pow(std::max((T)9.9999999999999998e-13, t3), 4);
    T t24 = std::max((T)0, t3);
    T t25 = t24/std::pow(t23, 1.0/2.0);
    T t26 = t25*t9;
    T t27 = t22*t26;
    T t28 = t27*std::fabs(t19);
    T t29 = t27*std::fabs(Q_plus[3]*n[0]);
    T t30 = std::pow(t4, 2);
    T t31 = std::pow(t24, 2)*t30/(std::pow(t20, 2)*t23);
    T t32 = p[0]*t2;
    T t33 = 2*t18;
    T t34 = std::pow(Q_minus[3], 2);
    T t35 = (2.0/3.0)*t18;
    T t36 = p[0]*t4;
    T t37 = 2*t31;
    T t38 = std::pow(Q_plus[3], 2);
    T t39 = (2.0/3.0)*t31;
    T t40 = Q_minus[2]*t33;
    T t41 = Q_minus[3]*Q_minus[5];
    T t42 = Q_plus[2]*t37;
    T t43 = Q_plus[3]*Q_plus[5];
    T t44 = Q_minus[2]*t8;
    T t45 = t13*t44;
    T t46 = Q_minus[3]*t8;
    T t47 = t13*t46;
    T t48 = (2.0/5.0)*t47;
    T t49 = Q_minus[5]*t8;
    T t50 = t13*t49;
    T t51 = Q_minus[4]*t8;
    T t52 = (4.0/5.0)*t18;
    T t53 = Qaux_minus[0]*t52;
    T t54 = Q_plus[2]*t22;
    T t55 = t26*t54;
    T t56 = Q_plus[3]*t22*t26;
    T t57 = (2.0/5.0)*t56;
    T t58 = Q_plus[5]*t22;
    T t59 = t26*t58;
    T t60 = Q_plus[4]*t22;
    T t61 = (4.0/5.0)*t31;
    T t62 = Qaux_plus[0]*t61;
    T t63 = 0.5*std::max(t5, std::max(t15 + 2*t16 + 4*t18*std::fabs(Q_minus[3]*t6), std::max(t28 + 2*t29 + 4*t31*std::fabs(Q_plus[3]*t19), std::max(2*t15 + (2.0/3.0)*t16 + t32*t5 + std::fabs(n[0]*(std::pow(Q_minus[2], 2)*t33 - t32 + t34*t35)), std::max(2*t28 + (2.0/3.0)*t29 + t36*t5 + std::fabs(n[0]*(std::pow(Q_plus[2], 2)*t37 - t36 + t38*t39)), std::max(t14*std::fabs(Q_minus[4]*n[0]) + (1.0/3.0)*t14*std::fabs(Q_minus[5]*n[0]) + t15 + (1.0/3.0)*t16 + std::fabs(n[0]*(Q_minus[4]*t40 + t35*t41)), std::max(t27*std::fabs(Q_plus[4]*n[0]) + (1.0/3.0)*t27*std::fabs(Q_plus[5]*n[0]) + t28 + (1.0/3.0)*t29 + std::fabs(n[0]*(Q_plus[4]*t42 + t39*t43)), std::max((3.0/5.0)*t16 + std::fabs(n[0]*(t45 + t48)) + std::fabs(n[0]*(Qaux_minus[0]*t48 + t50)) + std::fabs(n[0]*((2.0/5.0)*Qaux_minus[0]*t45 - 4.0/5.0*Qaux_minus[0]*t47 - 3.0/5.0*t13*t51 + (3.0/5.0)*t50)) + std::fabs(n[0]*(Q_minus[2]*Q_minus[3]*t53 + (6.0/5.0)*Q_minus[3]*Q_minus[4]*t18 + Q_minus[5]*t40 - t34*t53 + t41*t52)), (3.0/5.0)*t29 + std::fabs(n[0]*(t55 + t57)) + std::fabs(n[0]*(Qaux_plus[0]*t57 + t59)) + std::fabs(n[0]*((2.0/5.0)*Qaux_plus[0]*t55 - 4.0/5.0*Qaux_plus[0]*t56 - 3.0/5.0*t26*t60 + (3.0/5.0)*t59)) + std::fabs(n[0]*(Q_plus[2]*Q_plus[3]*t62 + (6.0/5.0)*Q_plus[3]*Q_plus[4]*t31 + Q_plus[5]*t42 - t38*t62 + t43*t61))))))))));
    T t64 = t63*(-t2 + t4);
    T t65 = (1.0/2.0)*p[0];
    T t66 = t63*(Q_plus[2]*t21*t4 - t44);
    T t67 = 0.8872983346207417*t9;
    T t68 = 0.1127016653792583*t9;
    T t69 = t12*t67 + t25*t68;
    T t70 = 0.27777777777777785*n[0];
    T t71 = 0.5*t9;
    T t72 = t12*t71 + t25*t71;
    T t73 = 0.44444444444444442*n[0];
    T t74 = t12*t68 + t25*t67;
    T t75 = Q_plus[3]*t21*t4 - t46;
    T t76 = 0.5*t75;
    T t77 = t76*(-t69*t70*(0.8872983346207417*t44 + 0.1127016653792583*t54) - t70*t74*(0.1127016653792583*t44 + 0.8872983346207417*t54) - t72*t73*(0.5*t44 + 0.5*t54));
    T t78 = t63*t75;
    T t79 = t63*(Q_plus[4]*t21*t4 - t51);
    T t80 = t76*(t70*((1.0/5.0)*t69*(0.8872983346207417*t49 + 0.1127016653792583*t58) - 6.0/5.0*t69*(0.8872983346207417*t51 + 0.1127016653792583*t60)) + t70*((1.0/5.0)*t74*(0.1127016653792583*t49 + 0.8872983346207417*t58) - 6.0/5.0*t74*(0.1127016653792583*t51 + 0.8872983346207417*t60)) + t73*((1.0/5.0)*t72*(0.5*t49 + 0.5*t58) - 6.0/5.0*t72*(0.5*t51 + 0.5*t60)));
    T t81 = t63*(Q_plus[5]*t21*t4 - t49);
    SimpleArray<T, 12> res;
    res[0] = 0;
    res[1] = t64;
    res[2] = n[0]*(-std::pow(Q_plus[1], 2)*t65 + (1.0/2.0)*p[0]*t30) + t66;
    res[3] = t77 + t78;
    res[4] = t79;
    res[5] = t80 + t81;
    res[6] = 0;
    res[7] = -t64;
    res[8] = n[0]*(std::pow(Q_minus[1], 2)*t65 - t17*t65) - t66;
    res[9] = t77 - t78;
    res[10] = -t79;
    res[11] = t80 - t81;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = n[0]*Q[2];
    T t1 = Qaux[3]*std::fabs(t0);
    T t2 = Qaux[3]*std::fabs(n[0]*Q[3]);
    T t3 = std::pow(Qaux[3], 2);
    T t4 = std::pow(Q[3], 2);
    T t5 = (1.0/3.0)*t3;
    T t6 = Q[2]*t3;
    T t7 = Q[3]*Q[5];
    T t8 = Qaux[3]*Q[2];
    T t9 = Qaux[3]*Q[3];
    T t10 = (2.0/5.0)*t9;
    T t11 = Qaux[3]*Q[5];
    T t12 = (3.0/5.0)*Q[4];
    T t13 = (2.0/5.0)*Qaux[0];
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(n[0]), std::max(t1 + 2*t2 + 2*t3*std::fabs(Q[3]*t0), std::max(Qaux[3]*std::fabs(n[0]*Q[4]) + (1.0/3.0)*Qaux[3]*std::fabs(n[0]*Q[5]) + t1 + (1.0/3.0)*t2 + std::fabs(n[0]*(Q[4]*t6 + t5*t7)), std::max(p[0]*std::fabs(Q[1]*n[0]) + 2*t1 + (2.0/3.0)*t2 + std::fabs(n[0]*(-p[0]*Q[1] + std::pow(Q[2], 2)*t3 + t4*t5)), (3.0/5.0)*t2 + std::fabs(n[0]*(t10 + t8)) + std::fabs(n[0]*(Qaux[0]*t10 + t11)) + std::fabs(n[0]*((4.0/5.0)*Qaux[0]*t9 + Qaux[3]*t12 - 3.0/5.0*t11 - t13*t8)) + std::fabs(n[0]*(Q[3]*t12*t3 + Q[3]*t13*t6 + Q[5]*t6 - t13*t3*t4 + (2.0/5.0)*t3*t7))))));
    return res;
    }

};
