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
    T t5 = p[0]*n[0]*t2;
    T t6 = std::pow(t2, 2);
    T t7 = std::pow(t2, 4) + std::pow(std::max((T)9.9999999999999998e-13, t1), 4);
    T t8 = (1.0 / std::pow(t7, 1));
    T t9 = std::max((T)0, t1);
    T t10 = std::pow(t9, 2);
    T t11 = std::max((T)1.0e-14, Q_minus[1]);
    T t12 = (1.0 / std::pow(t11, 2));
    T t13 = t10*t12*t6*t8;
    T t14 = 2*t13;
    T t15 = std::pow(Q_minus[3], 2);
    T t16 = (2.0/3.0)*t13;
    T t17 = n[0]*(-std::pow(Q_minus[2], 2)*t14 + p[0]*t2 - t15*t16);
    T t18 = t2/t11;
    T t19 = Q_minus[2]*t18;
    T t20 = std::pow((T)2, 1.0/2.0);
    T t21 = t9/std::pow(t7, 1.0/2.0);
    T t22 = t20*t21;
    T t23 = t19*t22;
    T t24 = n[0]*t23;
    T t25 = 2*t24;
    T t26 = Q_minus[3]*t18;
    T t27 = t22*t26;
    T t28 = n[0]*t27;
    T t29 = (2.0/3.0)*t28;
    T t30 = Q_minus[3]*t13;
    T t31 = Q_minus[2]*t30;
    T t32 = 4*n[0];
    T t33 = -t31*t32;
    T t34 = 2*t28;
    T t35 = Q_minus[2]*t14;
    T t36 = Q_minus[3]*Q_minus[5];
    T t37 = n[0]*(-Q_minus[4]*t35 - t16*t36);
    T t38 = Q_minus[4]*t18;
    T t39 = t22*t38;
    T t40 = n[0]*t39;
    T t41 = Q_minus[5]*t18;
    T t42 = t22*t41;
    T t43 = (1.0/3.0)*n[0];
    T t44 = t42*t43;
    T t45 = (1.0/3.0)*t28;
    T t46 = (4.0/5.0)*Qaux_minus[0];
    T t47 = n[0]*(-6.0/5.0*Q_minus[4]*t30 - Q_minus[5]*t35 + (4.0/5.0)*Qaux_minus[0]*t10*t12*t15*t6*t8 - 4.0/5.0*t13*t36 - t31*t46);
    T t48 = (2.0/5.0)*t27;
    T t49 = n[0]*(Qaux_minus[0]*t48 + t42);
    T t50 = n[0]*((2.0/5.0)*Qaux_minus[0]*t23 - t27*t46 - 3.0/5.0*t39 + (3.0/5.0)*t42);
    T t51 = (3.0/5.0)*t28;
    T t52 = n[0]*(t23 + t48);
    T t53 = p[0]*n[0]*t4;
    T t54 = std::pow(t4, 2);
    T t55 = std::pow(t4, 4) + std::pow(std::max((T)9.9999999999999998e-13, t3), 4);
    T t56 = (1.0 / std::pow(t55, 1));
    T t57 = std::max((T)0, t3);
    T t58 = std::pow(t57, 2);
    T t59 = std::max((T)1.0e-14, Q_plus[1]);
    T t60 = (1.0 / std::pow(t59, 2));
    T t61 = t54*t56*t58*t60;
    T t62 = 2*t61;
    T t63 = std::pow(Q_plus[3], 2);
    T t64 = (2.0/3.0)*t61;
    T t65 = n[0]*(-std::pow(Q_plus[2], 2)*t62 + p[0]*t4 - t63*t64);
    T t66 = (1.0 / std::pow(t59, 1));
    T t67 = t4*t66;
    T t68 = Q_plus[2]*t67;
    T t69 = t57/std::pow(t55, 1.0/2.0);
    T t70 = t20*t69;
    T t71 = t68*t70;
    T t72 = n[0]*t71;
    T t73 = 2*t72;
    T t74 = Q_plus[3]*t67*t70;
    T t75 = n[0]*t74;
    T t76 = (2.0/3.0)*t75;
    T t77 = Q_plus[3]*t61;
    T t78 = Q_plus[2]*t77;
    T t79 = -t32*t78;
    T t80 = 2*t75;
    T t81 = Q_plus[2]*t62;
    T t82 = Q_plus[3]*Q_plus[5];
    T t83 = n[0]*(-Q_plus[4]*t81 - t64*t82);
    T t84 = Q_plus[4]*t67;
    T t85 = t70*t84;
    T t86 = n[0]*t85;
    T t87 = Q_plus[5]*t67;
    T t88 = t70*t87;
    T t89 = t43*t88;
    T t90 = t43*t74;
    T t91 = (4.0/5.0)*Qaux_plus[0];
    T t92 = n[0]*(-6.0/5.0*Q_plus[4]*t77 - Q_plus[5]*t81 + (4.0/5.0)*Qaux_plus[0]*t54*t56*t58*t60*t63 - 4.0/5.0*t61*t82 - t78*t91);
    T t93 = (2.0/5.0)*t74;
    T t94 = n[0]*(Qaux_plus[0]*t93 + t88);
    T t95 = n[0]*((2.0/5.0)*Qaux_plus[0]*t71 - t74*t91 - 3.0/5.0*t85 + (3.0/5.0)*t88);
    T t96 = (3.0/5.0)*t75;
    T t97 = n[0]*(t71 + t93);
    T t98 = 0.5*std::max(std::fabs(eigenvalues(0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t5, t17, t25, t29, 0, 0, 0, t33, t34, t24, 0, 0, 0, t37, t40, t44, t24, t45, 0, t47, t49, t50, t51, t52)), std::max(std::fabs(eigenvalues(0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t53, t65, t73, t76, 0, 0, 0, t79, t80, t72, 0, 0, 0, t83, t86, t89, t72, t90, 0, t92, t94, t95, t96, t97)), std::max(std::fabs(eigenvalues(1, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t5, t17, t25, t29, 0, 0, 0, t33, t34, t24, 0, 0, 0, t37, t40, t44, t24, t45, 0, t47, t49, t50, t51, t52)), std::max(std::fabs(eigenvalues(1, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t53, t65, t73, t76, 0, 0, 0, t79, t80, t72, 0, 0, 0, t83, t86, t89, t72, t90, 0, t92, t94, t95, t96, t97)), std::max(std::fabs(eigenvalues(2, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t5, t17, t25, t29, 0, 0, 0, t33, t34, t24, 0, 0, 0, t37, t40, t44, t24, t45, 0, t47, t49, t50, t51, t52)), std::max(std::fabs(eigenvalues(2, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t53, t65, t73, t76, 0, 0, 0, t79, t80, t72, 0, 0, 0, t83, t86, t89, t72, t90, 0, t92, t94, t95, t96, t97)), std::max(std::fabs(eigenvalues(3, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t5, t17, t25, t29, 0, 0, 0, t33, t34, t24, 0, 0, 0, t37, t40, t44, t24, t45, 0, t47, t49, t50, t51, t52)), std::max(std::fabs(eigenvalues(3, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t53, t65, t73, t76, 0, 0, 0, t79, t80, t72, 0, 0, 0, t83, t86, t89, t72, t90, 0, t92, t94, t95, t96, t97)), std::max(std::fabs(eigenvalues(4, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t5, t17, t25, t29, 0, 0, 0, t33, t34, t24, 0, 0, 0, t37, t40, t44, t24, t45, 0, t47, t49, t50, t51, t52)), std::max(std::fabs(eigenvalues(4, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t53, t65, t73, t76, 0, 0, 0, t79, t80, t72, 0, 0, 0, t83, t86, t89, t72, t90, 0, t92, t94, t95, t96, t97)), std::max(std::fabs(eigenvalues(5, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t5, t17, t25, t29, 0, 0, 0, t33, t34, t24, 0, 0, 0, t37, t40, t44, t24, t45, 0, t47, t49, t50, t51, t52)), std::fabs(eigenvalues(5, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t53, t65, t73, t76, 0, 0, 0, t79, t80, t72, 0, 0, 0, t83, t86, t89, t72, t90, 0, t92, t94, t95, t96, t97)))))))))))));
    T t99 = t98*(-t2 + t4);
    T t100 = (1.0/2.0)*p[0];
    T t101 = t98*(Q_plus[2]*t4*t66 - t19);
    T t102 = 0.8872983346207417*t20;
    T t103 = 0.1127016653792583*t20;
    T t104 = t102*t21 + t103*t69;
    T t105 = 0.27777777777777785*n[0];
    T t106 = 0.5*t20;
    T t107 = t106*t21 + t106*t69;
    T t108 = 0.44444444444444442*n[0];
    T t109 = t102*t69 + t103*t21;
    T t110 = Q_plus[3]*t4*t66 - t26;
    T t111 = 0.5*t110;
    T t112 = t111*(-t104*t105*(0.8872983346207417*t19 + 0.1127016653792583*t68) - t105*t109*(0.1127016653792583*t19 + 0.8872983346207417*t68) - t107*t108*(0.5*t19 + 0.5*t68));
    T t113 = t110*t98;
    T t114 = t98*(Q_plus[4]*t4*t66 - t38);
    T t115 = t111*(t105*(-6.0/5.0*t104*(0.8872983346207417*t38 + 0.1127016653792583*t84) + (1.0/5.0)*t104*(0.8872983346207417*t41 + 0.1127016653792583*t87)) + t105*(-6.0/5.0*t109*(0.1127016653792583*t38 + 0.8872983346207417*t84) + (1.0/5.0)*t109*(0.1127016653792583*t41 + 0.8872983346207417*t87)) + t108*(-6.0/5.0*t107*(0.5*t38 + 0.5*t84) + (1.0/5.0)*t107*(0.5*t41 + 0.5*t87)));
    T t116 = t98*(Q_plus[5]*t4*t66 - t41);
    SimpleArray<T, 12> res;
    res[0] = 0;
    res[1] = t99;
    res[2] = n[0]*(-std::pow(Q_plus[1], 2)*t100 + (1.0/2.0)*p[0]*t54) + t101;
    res[3] = t112 + t113;
    res[4] = t114;
    res[5] = t115 + t116;
    res[6] = 0;
    res[7] = -t99;
    res[8] = n[0]*(std::pow(Q_minus[1], 2)*t100 - t100*t6) - t101;
    res[9] = t112 - t113;
    res[10] = -t114;
    res[11] = t115 - t116;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = p[0]*Q[1]*n[0];
    T t1 = std::pow(Qaux[3], 2);
    T t2 = std::pow(Q[3], 2);
    T t3 = (1.0/3.0)*t1;
    T t4 = n[0]*(p[0]*Q[1] - std::pow(Q[2], 2)*t1 - t2*t3);
    T t5 = Qaux[3]*Q[2];
    T t6 = n[0]*t5;
    T t7 = 2*t6;
    T t8 = Qaux[3]*Q[3];
    T t9 = n[0]*t8;
    T t10 = (2.0/3.0)*t9;
    T t11 = Q[2]*t1;
    T t12 = Q[3]*t11;
    T t13 = -2*n[0]*t12;
    T t14 = 2*t9;
    T t15 = Q[3]*Q[5];
    T t16 = n[0]*(-Q[4]*t11 - t15*t3);
    T t17 = Qaux[3]*Q[4];
    T t18 = n[0]*t17;
    T t19 = Qaux[3]*Q[5];
    T t20 = (1.0/3.0)*n[0]*t19;
    T t21 = (1.0/3.0)*t9;
    T t22 = (2.0/5.0)*Qaux[0];
    T t23 = n[0]*((2.0/5.0)*Qaux[0]*t1*t2 - 3.0/5.0*Q[3]*Q[4]*t1 - Q[5]*t11 - 2.0/5.0*t1*t15 - t12*t22);
    T t24 = (2.0/5.0)*t8;
    T t25 = n[0]*(Qaux[0]*t24 + t19);
    T t26 = n[0]*(-4.0/5.0*Qaux[0]*t8 - 3.0/5.0*t17 + (3.0/5.0)*t19 + t22*t5);
    T t27 = (3.0/5.0)*t9;
    T t28 = n[0]*(t24 + t5);
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(eigenvalues(0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t0, t4, t7, t10, 0, 0, 0, t13, t14, t6, 0, 0, 0, t16, t18, t20, t6, t21, 0, t23, t25, t26, t27, t28)), std::max(std::fabs(eigenvalues(1, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t0, t4, t7, t10, 0, 0, 0, t13, t14, t6, 0, 0, 0, t16, t18, t20, t6, t21, 0, t23, t25, t26, t27, t28)), std::max(std::fabs(eigenvalues(2, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t0, t4, t7, t10, 0, 0, 0, t13, t14, t6, 0, 0, 0, t16, t18, t20, t6, t21, 0, t23, t25, t26, t27, t28)), std::max(std::fabs(eigenvalues(3, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t0, t4, t7, t10, 0, 0, 0, t13, t14, t6, 0, 0, 0, t16, t18, t20, t6, t21, 0, t23, t25, t26, t27, t28)), std::max(std::fabs(eigenvalues(4, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t0, t4, t7, t10, 0, 0, 0, t13, t14, t6, 0, 0, 0, t16, t18, t20, t6, t21, 0, t23, t25, t26, t27, t28)), std::fabs(eigenvalues(5, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, 0, 0, t0, t4, t7, t10, 0, 0, 0, t13, t14, t6, 0, 0, 0, t16, t18, t20, t6, t21, 0, t23, t25, t26, t27, t28)))))));
    return res;
    }

};
