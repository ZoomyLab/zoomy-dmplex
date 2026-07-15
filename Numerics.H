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
    static constexpr int n_dof_q = 8;
    PORTABLE_FN static inline SimpleArray<T, 8> numerical_flux(
        const T* Q_minus,
        const T* Q_plus,
        const T* Qaux_minus,
        const T* Qaux_plus,
        const T* p,
        const T* n)
    {
    T t0 = 0.5*n[0];
    T t1 = std::max((T)1.0e-14, Q_minus[1]);
    T t2 = -std::max(Q_minus[0], Q_plus[0]);
    T t3 = Q_minus[0] + Q_minus[1] + t2;
    T t4 = std::max((T)0.0, t3);
    T t5 = t4/t1;
    T t6 = 0.5*n[1];
    T t7 = std::max((T)1.0e-14, Q_plus[1]);
    T t8 = Q_plus[0] + Q_plus[1] + t2;
    T t9 = std::max((T)0.0, t8);
    T t10 = t9/t7;
    T t11 = std::pow(t4, 2);
    T t12 = std::pow((T)2, 1.0/2.0);
    T t13 = t11*t12*std::max((T)0, t3)/(std::pow(t1, 2)*std::pow(std::pow(t4, 4) + std::pow(std::max((T)9.9999999999999998e-13, t3), 4), 1.0/2.0));
    T t14 = Q_minus[2]*t13;
    T t15 = (1.0/3.0)*t13;
    T t16 = Q_minus[3]*t15;
    T t17 = Q_minus[4]*t14 + Q_minus[5]*t16;
    T t18 = std::pow(t9, 2);
    T t19 = t12*t18*std::max((T)0, t8)/(std::pow(t7, 2)*std::pow(std::pow(t9, 4) + std::pow(std::max((T)9.9999999999999998e-13, t8), 4), 1.0/2.0));
    T t20 = Q_plus[2]*t19;
    T t21 = (1.0/3.0)*t19;
    T t22 = Q_plus[3]*t21;
    T t23 = Q_plus[4]*t20 + Q_plus[5]*t22;
    T t24 = (1.0/2.0)*p[0];
    T t25 = t11*t24;
    T t26 = std::pow(Q_minus[3], 2);
    T t27 = t18*t24;
    T t28 = std::pow(Q_plus[3], 2);
    T t29 = 1.0*n[0];
    T t30 = Q_minus[3]*t14;
    T t31 = Q_plus[3]*t20;
    T t32 = Q_minus[5]*t14;
    T t33 = Q_minus[4]*t13;
    T t34 = Q_minus[3]*t33;
    T t35 = t32 + t34;
    T t36 = Q_plus[5]*t20;
    T t37 = Q_plus[4]*t19;
    T t38 = Q_plus[3]*t37;
    T t39 = t36 + t38;
    T t40 = std::pow(Q_minus[5], 2);
    T t41 = std::pow(Q_plus[5], 2);
    T t42 = 1.0*n[1];
    T t43 = Q_minus[5]*t33;
    T t44 = Q_plus[5]*t37;
    T t45 = Q_minus[5]*Q_minus[7];
    T t46 = Q_plus[5]*Q_plus[7];
    T t47 = (3.0/5.0)*Q_minus[6]*t13;
    T t48 = (2.0/5.0)*t13;
    T t49 = Q_minus[3]*t48;
    T t50 = (2.0/5.0)*Qaux_minus[0];
    T t51 = (2.0/5.0)*Qaux_minus[1];
    T t52 = Qaux_minus[0]*t48;
    T t53 = (3.0/5.0)*Q_plus[6]*t19;
    T t54 = (2.0/5.0)*t19;
    T t55 = Q_plus[3]*t54;
    T t56 = (2.0/5.0)*Qaux_plus[0];
    T t57 = (2.0/5.0)*Qaux_plus[1];
    T t58 = Qaux_plus[0]*t54;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = Q_minus[2]*t0*t5 + Q_minus[4]*t5*t6 + Q_plus[2]*t0*t10 + Q_plus[4]*t10*t6;
    res[2] = t0*(std::pow(Q_minus[2], 2)*t13 + t15*t26 + t25) + t0*(std::pow(Q_plus[2], 2)*t19 + t21*t28 + t27) + t17*t6 + t23*t6;
    res[3] = t29*t30 + t29*t31 + t35*t6 + t39*t6;
    res[4] = t0*t17 + t0*t23 + t6*(std::pow(Q_minus[4], 2)*t13 + t15*t40 + t25) + t6*(std::pow(Q_plus[4], 2)*t19 + t21*t41 + t27);
    res[5] = t0*t35 + t0*t39 + t42*t43 + t42*t44;
    res[6] = t0*(Q_minus[6]*t14 + Q_minus[7]*t16) + t0*(Q_plus[6]*t20 + Q_plus[7]*t22) + t6*(Q_minus[6]*t33 + t15*t45) + t6*(Q_plus[6]*t37 + t21*t46);
    res[7] = t0*(Q_minus[3]*t47 - Q_minus[5]*Qaux_minus[1]*t49 + Q_minus[7]*t14 + Q_minus[7]*t49 - t26*t52 + t30*t50 + t34*t51) + t0*(Q_plus[3]*t53 - Q_plus[5]*Qaux_plus[1]*t55 + Q_plus[7]*t20 + Q_plus[7]*t55 - t28*t58 + t31*t56 + t38*t57) + t6*(-Q_minus[3]*Q_minus[5]*t52 + Q_minus[5]*t47 + Q_minus[7]*t33 - Qaux_minus[1]*t40*t48 + t32*t50 + t43*t51 + t45*t48) + t6*(-Q_plus[3]*Q_plus[5]*t58 + Q_plus[5]*t53 + Q_plus[7]*t37 - Qaux_plus[1]*t41*t54 + t36*t56 + t44*t57 + t46*t54);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> numerical_fluctuations(
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
    T t6 = std::fabs(n[1]);
    T t7 = std::max((T)1.0e-14, Q_minus[1]);
    T t8 = t2/t7;
    T t9 = std::pow((T)2, 1.0/2.0);
    T t10 = std::pow(t2, 4) + std::pow(std::max((T)9.9999999999999998e-13, t1), 4);
    T t11 = std::max((T)0, t1);
    T t12 = t11/std::pow(t10, 1.0/2.0);
    T t13 = t12*t9;
    T t14 = t13*t8;
    T t15 = t14*std::fabs(Q_minus[5]*n[0]);
    T t16 = Q_minus[2]*t8;
    T t17 = t13*t16;
    T t18 = n[0]*t17;
    T t19 = Q_minus[4]*t8;
    T t20 = t13*t19;
    T t21 = n[1]*t20;
    T t22 = std::fabs(t18 + t21);
    T t23 = Q_minus[3]*t8;
    T t24 = t13*t23;
    T t25 = n[0]*t24;
    T t26 = Q_minus[5]*t8;
    T t27 = t13*t26;
    T t28 = n[1]*t27;
    T t29 = std::pow(t2, 2);
    T t30 = std::pow(t11, 2)*t29/(t10*std::pow(t7, 2));
    T t31 = Q_minus[5]*t30;
    T t32 = 4*n[1];
    T t33 = 2*t30;
    T t34 = Q_minus[2]*t33;
    T t35 = Q_minus[4]*t33;
    T t36 = Q_minus[3]*t35 + Q_minus[5]*t34;
    T t37 = t14*std::fabs(Q_minus[3]*n[1]);
    T t38 = Q_minus[3]*t30;
    T t39 = 4*n[0];
    T t40 = std::max((T)1.0e-14, Q_plus[1]);
    T t41 = (1.0 / std::pow(t40, 1));
    T t42 = t4*t41;
    T t43 = std::pow(t4, 4) + std::pow(std::max((T)9.9999999999999998e-13, t3), 4);
    T t44 = std::max((T)0, t3);
    T t45 = t44/std::pow(t43, 1.0/2.0);
    T t46 = t45*t9;
    T t47 = t42*t46;
    T t48 = t47*std::fabs(Q_plus[5]*n[0]);
    T t49 = Q_plus[2]*t42;
    T t50 = t46*t49;
    T t51 = n[0]*t50;
    T t52 = Q_plus[4]*t42;
    T t53 = t46*t52;
    T t54 = n[1]*t53;
    T t55 = std::fabs(t51 + t54);
    T t56 = Q_plus[3]*t42*t46;
    T t57 = n[0]*t56;
    T t58 = Q_plus[5]*t42*t46;
    T t59 = n[1]*t58;
    T t60 = std::pow(t4, 2);
    T t61 = std::pow(t44, 2)*t60/(std::pow(t40, 2)*t43);
    T t62 = Q_plus[5]*t61;
    T t63 = 2*t61;
    T t64 = Q_plus[2]*t63;
    T t65 = Q_plus[4]*t63;
    T t66 = Q_plus[3]*t65 + Q_plus[5]*t64;
    T t67 = t47*std::fabs(Q_plus[3]*n[1]);
    T t68 = Q_plus[3]*t61;
    T t69 = p[0]*t2;
    T t70 = (1.0/3.0)*t28;
    T t71 = (2.0/3.0)*t30;
    T t72 = Q_minus[3]*t71;
    T t73 = Q_minus[4]*t34 + Q_minus[5]*t72;
    T t74 = -t69;
    T t75 = std::pow(Q_minus[3], 2);
    T t76 = p[0]*t4;
    T t77 = (1.0/3.0)*t59;
    T t78 = (2.0/3.0)*t61;
    T t79 = Q_plus[3]*t78;
    T t80 = Q_plus[4]*t64 + Q_plus[5]*t79;
    T t81 = -t76;
    T t82 = std::pow(Q_plus[3], 2);
    T t83 = (1.0/3.0)*t25;
    T t84 = std::pow(Q_minus[5], 2);
    T t85 = (1.0/3.0)*t57;
    T t86 = std::pow(Q_plus[5], 2);
    T t87 = (1.0/3.0)*t14;
    T t88 = Q_minus[5]*Q_minus[7];
    T t89 = (1.0/3.0)*t47;
    T t90 = Q_plus[5]*Q_plus[7];
    T t91 = (2.0/5.0)*Qaux_minus[1];
    T t92 = t25*t91;
    T t93 = Q_minus[7]*t8;
    T t94 = t13*t93;
    T t95 = (2.0/5.0)*t27;
    T t96 = Qaux_minus[1]*t95;
    T t97 = (2.0/5.0)*Qaux_minus[0];
    T t98 = t28*t97;
    T t99 = (2.0/5.0)*t24;
    T t100 = Qaux_minus[0]*t99;
    T t101 = Q_minus[6]*t8;
    T t102 = -3.0/5.0*t101*t13 + t17*t97 + t20*t91 + (3.0/5.0)*t94;
    T t103 = (6.0/5.0)*Q_minus[6];
    T t104 = (4.0/5.0)*t30;
    T t105 = Q_minus[3]*t104;
    T t106 = Qaux_minus[0]*t104;
    T t107 = Q_minus[3]*t106;
    T t108 = Qaux_minus[1]*t105;
    T t109 = Qaux_minus[1]*t104;
    T t110 = (2.0/5.0)*Qaux_plus[1];
    T t111 = t110*t57;
    T t112 = Q_plus[7]*t42;
    T t113 = t112*t46;
    T t114 = (2.0/5.0)*t58;
    T t115 = Qaux_plus[1]*t114;
    T t116 = (2.0/5.0)*Qaux_plus[0];
    T t117 = t116*t59;
    T t118 = (2.0/5.0)*t56;
    T t119 = Qaux_plus[0]*t118;
    T t120 = Q_plus[6]*t42;
    T t121 = t110*t53 + (3.0/5.0)*t113 + t116*t50 - 3.0/5.0*t120*t46;
    T t122 = (6.0/5.0)*Q_plus[6];
    T t123 = (4.0/5.0)*t61;
    T t124 = Q_plus[3]*t123;
    T t125 = Qaux_plus[0]*t123;
    T t126 = Q_plus[3]*t125;
    T t127 = Qaux_plus[1]*t124;
    T t128 = Qaux_plus[1]*t123;
    T t129 = 0.5*std::max(t5 + t6, std::max(t15 + t22 + std::fabs(t25 + 2*t28) + std::fabs(Q_minus[4]*t31*t32 + n[0]*t36), std::max(t22 + t37 + std::fabs(2*t25 + t28) + std::fabs(Q_minus[2]*t38*t39 + n[1]*t36), std::max(t48 + t55 + std::fabs(t57 + 2*t59) + std::fabs(Q_plus[4]*t32*t62 + n[0]*t66), std::max(t55 + t67 + std::fabs(2*t57 + t59) + std::fabs(Q_plus[2]*t39*t68 + n[1]*t66), std::max(t14*std::fabs(Q_minus[2]*n[1]) + (1.0/3.0)*t37 + t5*t69 + std::fabs(2*t18 + t21) + std::fabs((2.0/3.0)*t25 + t70) + std::fabs(n[0]*(std::pow(Q_minus[2], 2)*t33 + t71*t75 + t74) + n[1]*t73), std::max(t14*std::fabs(Q_minus[4]*n[0]) + (1.0/3.0)*t15 + t6*t69 + std::fabs(t18 + 2*t21) + std::fabs((2.0/3.0)*t28 + t83) + std::fabs(n[0]*t73 + n[1]*(std::pow(Q_minus[4], 2)*t33 + t71*t84 + t74)), std::max(t47*std::fabs(Q_plus[2]*n[1]) + t5*t76 + (1.0/3.0)*t67 + std::fabs(2*t51 + t54) + std::fabs((2.0/3.0)*t57 + t77) + std::fabs(n[0]*(std::pow(Q_plus[2], 2)*t63 + t78*t82 + t81) + n[1]*t80), std::max(t47*std::fabs(Q_plus[4]*n[0]) + (1.0/3.0)*t48 + t6*t76 + std::fabs(t51 + 2*t54) + std::fabs((2.0/3.0)*t59 + t85) + std::fabs(n[0]*t80 + n[1]*(std::pow(Q_plus[4], 2)*t63 + t78*t86 + t81)), std::max(t14*std::fabs(Q_minus[6]*n[0]) + t14*std::fabs(Q_minus[6]*n[1]) + t22 + t87*std::fabs(Q_minus[7]*n[0]) + t87*std::fabs(Q_minus[7]*n[1]) + std::fabs(t70 + t83) + std::fabs(n[0]*(Q_minus[6]*t34 + Q_minus[7]*t72) + n[1]*(Q_minus[6]*t35 + t71*t88)), std::max(t47*std::fabs(Q_plus[6]*n[0]) + t47*std::fabs(Q_plus[6]*n[1]) + t55 + t89*std::fabs(Q_plus[7]*n[0]) + t89*std::fabs(Q_plus[7]*n[1]) + std::fabs(t77 + t85) + std::fabs(n[0]*(Q_plus[6]*t64 + Q_plus[7]*t79) + n[1]*(Q_plus[6]*t65 + t78*t90)), std::max(std::fabs((3.0/5.0)*t25 + (3.0/5.0)*t28) + std::fabs(n[0]*(t100 + t94) + t98) + std::fabs(n[0]*(t17 + t99) + n[1]*(t20 + t95)) + std::fabs(-n[0]*(-4.0/5.0*Qaux_minus[0]*t24 + t102 - t96) + t98) + std::fabs(n[0]*(Q_minus[2]*t107 + Q_minus[4]*t108 - Q_minus[5]*t108 + Q_minus[7]*t105 + Q_minus[7]*t34 + t103*t38 - t106*t75) + n[1]*(Q_minus[2]*Q_minus[5]*t106 + Q_minus[4]*Q_minus[5]*t109 - Q_minus[5]*t107 + Q_minus[7]*t35 + t103*t31 + t104*t88 - t109*t84)) + std::fabs(n[1]*(t94 + t96) + t92) + std::fabs(-n[1]*(-4.0/5.0*Qaux_minus[1]*t27 - t100 + t102) + t92), std::fabs((3.0/5.0)*t57 + (3.0/5.0)*t59) + std::fabs(n[0]*(t113 + t119) + t117) + std::fabs(n[0]*(t118 + t50) + n[1]*(t114 + t53)) + std::fabs(-n[0]*(-4.0/5.0*Qaux_plus[0]*t56 - t115 + t121) + t117) + std::fabs(n[0]*(Q_plus[2]*t126 + Q_plus[4]*t127 - Q_plus[5]*t127 + Q_plus[7]*t124 + Q_plus[7]*t64 + t122*t68 - t125*t82) + n[1]*(Q_plus[2]*Q_plus[5]*t125 + Q_plus[4]*Q_plus[5]*t128 - Q_plus[5]*t126 + Q_plus[7]*t65 + t122*t62 + t123*t90 - t128*t86)) + std::fabs(n[1]*(t113 + t115) + t111) + std::fabs(-n[1]*(-4.0/5.0*Qaux_plus[1]*t58 - t119 + t121) + t111)))))))))))));
    T t130 = t129*(-t2 + t4);
    T t131 = (1.0/2.0)*p[0];
    T t132 = -std::pow(Q_plus[1], 2)*t131 + (1.0/2.0)*p[0]*t60;
    T t133 = t129*(Q_plus[2]*t4*t41 - t16);
    T t134 = Q_plus[3]*t4*t41 - t23;
    T t135 = t129*t134;
    T t136 = 0.27777777777777785*n[0];
    T t137 = 0.8872983346207417*t9;
    T t138 = 0.1127016653792583*t9;
    T t139 = t12*t137 + t138*t45;
    T t140 = t139*(0.8872983346207417*t16 + 0.1127016653792583*t49);
    T t141 = 0.5*t9;
    T t142 = t12*t141 + t141*t45;
    T t143 = 0.44444444444444442*t142;
    T t144 = t143*(0.5*t16 + 0.5*t49);
    T t145 = t12*t138 + t137*t45;
    T t146 = t145*(0.1127016653792583*t16 + 0.8872983346207417*t49);
    T t147 = 0.5*t134;
    T t148 = 0.27777777777777785*n[1];
    T t149 = Q_plus[5]*t4*t41 - t26;
    T t150 = 0.5*t149;
    T t151 = t147*(-n[0]*t144 - t136*t140 - t136*t146) + t150*(-n[1]*t144 - t140*t148 - t146*t148);
    T t152 = t129*(Q_plus[4]*t4*t41 - t19);
    T t153 = t129*t149;
    T t154 = t139*(0.8872983346207417*t19 + 0.1127016653792583*t52);
    T t155 = t143*(0.5*t19 + 0.5*t52);
    T t156 = t145*(0.1127016653792583*t19 + 0.8872983346207417*t52);
    T t157 = t147*(-n[0]*t155 - t136*t154 - t136*t156) + t150*(-n[1]*t155 - t148*t154 - t148*t156);
    T t158 = t129*(Q_plus[6]*t4*t41 - t101);
    T t159 = t129*(Q_plus[7]*t4*t41 - t93);
    T t160 = -6.0/5.0*t139*(0.8872983346207417*t101 + 0.1127016653792583*t120) + (1.0/5.0)*t139*(0.1127016653792583*t112 + 0.8872983346207417*t93);
    T t161 = -0.53333333333333333*t142*(0.5*t101 + 0.5*t120) + 0.088888888888888892*t142*(0.5*t112 + 0.5*t93);
    T t162 = -6.0/5.0*t145*(0.1127016653792583*t101 + 0.8872983346207417*t120) + (1.0/5.0)*t145*(0.8872983346207417*t112 + 0.1127016653792583*t93);
    T t163 = t147*(n[0]*t161 + t136*t160 + t136*t162) + t150*(n[1]*t161 + t148*t160 + t148*t162);
    T t164 = std::pow(Q_minus[1], 2)*t131 - t131*t29;
    SimpleArray<T, 16> res;
    res[0] = 0;
    res[1] = t130;
    res[2] = n[0]*t132 + t133;
    res[3] = t135 + t151;
    res[4] = n[1]*t132 + t152;
    res[5] = t153 + t157;
    res[6] = t158;
    res[7] = t159 + t163;
    res[8] = 0;
    res[9] = -t130;
    res[10] = n[0]*t164 - t133;
    res[11] = -t135 + t151;
    res[12] = n[1]*t164 - t152;
    res[13] = -t153 + t157;
    res[14] = -t158;
    res[15] = -t159 + t163;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = Qaux[6]*std::fabs(n[0]*Q[5]);
    T t1 = Qaux[6]*Q[2];
    T t2 = n[0]*t1;
    T t3 = Qaux[6]*Q[4];
    T t4 = n[1]*t3;
    T t5 = std::fabs(t2 + t4);
    T t6 = Qaux[6]*Q[3];
    T t7 = n[0]*t6;
    T t8 = Qaux[6]*Q[5];
    T t9 = n[1]*t8;
    T t10 = std::pow(Qaux[6], 2);
    T t11 = Q[4]*t10;
    T t12 = Q[5]*t11;
    T t13 = Q[2]*t10;
    T t14 = Q[5]*t13;
    T t15 = Q[3]*t11;
    T t16 = t14 + t15;
    T t17 = Qaux[6]*std::fabs(n[1]*Q[3]);
    T t18 = Q[3]*t13;
    T t19 = (1.0/3.0)*t9;
    T t20 = (1.0/3.0)*t10;
    T t21 = Q[3]*t20;
    T t22 = Q[4]*t13 + Q[5]*t21;
    T t23 = -p[0]*Q[1];
    T t24 = std::pow(Q[3], 2);
    T t25 = (1.0/3.0)*t7;
    T t26 = std::pow(Q[5], 2);
    T t27 = (1.0/3.0)*Qaux[6];
    T t28 = Q[5]*Q[7];
    T t29 = (2.0/5.0)*Qaux[0];
    T t30 = t29*t9;
    T t31 = Qaux[6]*Q[7];
    T t32 = (2.0/5.0)*t6;
    T t33 = Qaux[0]*t32;
    T t34 = (2.0/5.0)*Qaux[1];
    T t35 = t34*t7;
    T t36 = (2.0/5.0)*t8;
    T t37 = Qaux[1]*t36;
    T t38 = (3.0/5.0)*Q[6];
    T t39 = -Qaux[6]*t38 + t1*t29 + t3*t34 + (3.0/5.0)*t31;
    T t40 = t10*t38;
    T t41 = (2.0/5.0)*t10;
    T t42 = Q[3]*t41;
    T t43 = Qaux[0]*t41;
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(n[0]) + std::fabs(n[1]), std::max(t0 + t5 + std::fabs(t7 + 2*t9) + std::fabs(n[0]*t16 + 2*n[1]*t12), std::max(t17 + t5 + std::fabs(2*t7 + t9) + std::fabs(2*n[0]*t18 + n[1]*t16), std::max(p[0]*std::fabs(Q[1]*n[0]) + Qaux[6]*std::fabs(n[1]*Q[2]) + (1.0/3.0)*t17 + std::fabs(t19 + (2.0/3.0)*t7) + std::fabs(2*t2 + t4) + std::fabs(n[0]*(std::pow(Q[2], 2)*t10 + t20*t24 + t23) + n[1]*t22), std::max(p[0]*std::fabs(Q[1]*n[1]) + Qaux[6]*std::fabs(n[0]*Q[4]) + (1.0/3.0)*t0 + std::fabs(t2 + 2*t4) + std::fabs(t25 + (2.0/3.0)*t9) + std::fabs(n[0]*t22 + n[1]*(std::pow(Q[4], 2)*t10 + t20*t26 + t23)), std::max(Qaux[6]*std::fabs(n[0]*Q[6]) + Qaux[6]*std::fabs(n[1]*Q[6]) + t27*std::fabs(n[0]*Q[7]) + t27*std::fabs(n[1]*Q[7]) + t5 + std::fabs(t19 + t25) + std::fabs(n[0]*(Q[6]*t13 + Q[7]*t21) + n[1]*(Q[6]*t11 + t20*t28)), std::fabs((3.0/5.0)*t7 + (3.0/5.0)*t9) + std::fabs(n[0]*(t1 + t32) + n[1]*(t3 + t36)) + std::fabs(n[0]*(t31 + t33) + t30) + std::fabs(-n[0]*(-4.0/5.0*Qaux[0]*t6 - t37 + t39) + t30) + std::fabs(n[0]*(-Qaux[1]*Q[5]*t42 + Q[3]*t40 + Q[7]*t13 + Q[7]*t42 + t15*t34 + t18*t29 - t24*t43) + n[1]*(-Qaux[1]*t26*t41 - Q[3]*Q[5]*t43 + Q[5]*t40 + Q[7]*t11 + t12*t34 + t14*t29 + t28*t41)) + std::fabs(n[1]*(t31 + t37) + t35) + std::fabs(-n[1]*(-4.0/5.0*Qaux[1]*t8 - t33 + t39) + t35)))))));
    return res;
    }

};
