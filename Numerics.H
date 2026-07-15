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
    T t5 = p[0]*t2;
    T t6 = n[0]*t5;
    T t7 = std::pow(t2, 2);
    T t8 = std::pow(t2, 4) + std::pow(std::max((T)9.9999999999999998e-13, t1), 4);
    T t9 = (1.0 / std::pow(t8, 1));
    T t10 = std::max((T)0, t1);
    T t11 = std::pow(t10, 2);
    T t12 = std::max((T)1.0e-14, Q_minus[1]);
    T t13 = (1.0 / std::pow(t12, 2));
    T t14 = t11*t13*t7*t9;
    T t15 = 2*t14;
    T t16 = Q_minus[2]*t15;
    T t17 = (2.0/3.0)*t14;
    T t18 = Q_minus[3]*t17;
    T t19 = -Q_minus[4]*t16 - Q_minus[5]*t18;
    T t20 = -p[0]*t2;
    T t21 = std::pow(Q_minus[3], 2);
    T t22 = n[0]*(-std::pow(Q_minus[2], 2)*t15 - t17*t21 - t20) + n[1]*t19;
    T t23 = t2/t12;
    T t24 = Q_minus[4]*t23;
    T t25 = std::pow((T)2, 1.0/2.0);
    T t26 = t10/std::pow(t8, 1.0/2.0);
    T t27 = t25*t26;
    T t28 = t24*t27;
    T t29 = n[1]*t28;
    T t30 = Q_minus[2]*t23;
    T t31 = t27*t30;
    T t32 = n[0]*t31;
    T t33 = t29 + 2*t32;
    T t34 = Q_minus[3]*t23;
    T t35 = n[0]*t27;
    T t36 = t34*t35;
    T t37 = Q_minus[5]*t23;
    T t38 = n[1]*t27;
    T t39 = t37*t38;
    T t40 = (1.0/3.0)*t39;
    T t41 = (2.0/3.0)*t36 + t40;
    T t42 = n[1]*t31;
    T t43 = t34*t38;
    T t44 = (1.0/3.0)*t43;
    T t45 = Q_minus[3]*t14;
    T t46 = 4*n[0];
    T t47 = Q_minus[4]*t15;
    T t48 = -Q_minus[3]*t47 - Q_minus[5]*t16;
    T t49 = -Q_minus[2]*t45*t46 + n[1]*t48;
    T t50 = 2*t36 + t39;
    T t51 = t29 + t32;
    T t52 = n[1]*t5;
    T t53 = std::pow(Q_minus[5], 2);
    T t54 = n[0]*t19 + n[1]*(-std::pow(Q_minus[4], 2)*t15 - t17*t53 - t20);
    T t55 = n[0]*t28;
    T t56 = t35*t37;
    T t57 = (1.0/3.0)*t56;
    T t58 = 2*t29 + t32;
    T t59 = (1.0/3.0)*t36;
    T t60 = (2.0/3.0)*t39 + t59;
    T t61 = Q_minus[5]*t14;
    T t62 = 4*n[1];
    T t63 = -Q_minus[4]*t61*t62 + n[0]*t48;
    T t64 = t36 + 2*t39;
    T t65 = Q_minus[5]*Q_minus[7];
    T t66 = n[0]*(-Q_minus[6]*t16 - Q_minus[7]*t18) + n[1]*(-Q_minus[6]*t47 - t17*t65);
    T t67 = Q_minus[6]*t23;
    T t68 = t35*t67;
    T t69 = Q_minus[7]*t23;
    T t70 = t27*t69;
    T t71 = (1.0/3.0)*t70;
    T t72 = n[0]*t71;
    T t73 = t38*t67;
    T t74 = n[1]*t71;
    T t75 = t40 + t59;
    T t76 = (6.0/5.0)*Q_minus[6];
    T t77 = (4.0/5.0)*t45;
    T t78 = (4.0/5.0)*t14;
    T t79 = n[0]*(-Q_minus[2]*Qaux_minus[0]*t77 + (4.0/5.0)*Q_minus[3]*Q_minus[5]*Qaux_minus[1]*t11*t13*t7*t9 - Q_minus[4]*Qaux_minus[1]*t77 - Q_minus[7]*t16 - Q_minus[7]*t77 + (4.0/5.0)*Qaux_minus[0]*t11*t13*t21*t7*t9 - t45*t76) + n[1]*(-Q_minus[2]*Q_minus[5]*Qaux_minus[0]*t78 + (4.0/5.0)*Q_minus[3]*Q_minus[5]*Qaux_minus[0]*t11*t13*t7*t9 - Q_minus[4]*Q_minus[5]*Qaux_minus[1]*t78 - Q_minus[7]*t47 + (4.0/5.0)*Qaux_minus[1]*t11*t13*t53*t7*t9 - t61*t76 - t65*t78);
    T t80 = (2.0/5.0)*Qaux_minus[0];
    T t81 = t39*t80;
    T t82 = (2.0/5.0)*t27;
    T t83 = t34*t82;
    T t84 = Qaux_minus[0]*t83;
    T t85 = n[0]*(t70 + t84) + t81;
    T t86 = (4.0/5.0)*t27;
    T t87 = t37*t82;
    T t88 = Qaux_minus[1]*t87;
    T t89 = (2.0/5.0)*Qaux_minus[1];
    T t90 = -3.0/5.0*t27*t67 + t28*t89 + t31*t80 + (3.0/5.0)*t70;
    T t91 = n[0]*(-Qaux_minus[0]*t34*t86 - t88 + t90) - t81;
    T t92 = t36*t89;
    T t93 = n[1]*(t70 + t88) + t92;
    T t94 = n[1]*(-Qaux_minus[1]*t37*t86 - t84 + t90) - t92;
    T t95 = (3.0/5.0)*t36 + (3.0/5.0)*t39;
    T t96 = n[0]*(t31 + t83) + n[1]*(t28 + t87);
    T t97 = p[0]*t4;
    T t98 = n[0]*t97;
    T t99 = std::pow(t4, 2);
    T t100 = std::pow(t4, 4) + std::pow(std::max((T)9.9999999999999998e-13, t3), 4);
    T t101 = (1.0 / std::pow(t100, 1));
    T t102 = std::max((T)0, t3);
    T t103 = std::pow(t102, 2);
    T t104 = std::max((T)1.0e-14, Q_plus[1]);
    T t105 = (1.0 / std::pow(t104, 2));
    T t106 = t101*t103*t105*t99;
    T t107 = 2*t106;
    T t108 = Q_plus[2]*t107;
    T t109 = (2.0/3.0)*t106;
    T t110 = Q_plus[3]*t109;
    T t111 = -Q_plus[4]*t108 - Q_plus[5]*t110;
    T t112 = -p[0]*t4;
    T t113 = std::pow(Q_plus[3], 2);
    T t114 = n[0]*(-std::pow(Q_plus[2], 2)*t107 - t109*t113 - t112) + n[1]*t111;
    T t115 = (1.0 / std::pow(t104, 1));
    T t116 = t115*t4;
    T t117 = Q_plus[4]*t116;
    T t118 = t102/std::pow(t100, 1.0/2.0);
    T t119 = t118*t25;
    T t120 = t117*t119;
    T t121 = n[1]*t120;
    T t122 = Q_plus[2]*t116;
    T t123 = t119*t122;
    T t124 = n[0]*t123;
    T t125 = t121 + 2*t124;
    T t126 = Q_plus[3]*t116;
    T t127 = n[0]*t119;
    T t128 = t126*t127;
    T t129 = Q_plus[5]*t116;
    T t130 = n[1]*t119;
    T t131 = t129*t130;
    T t132 = (1.0/3.0)*t131;
    T t133 = (2.0/3.0)*t128 + t132;
    T t134 = n[1]*t123;
    T t135 = t126*t130;
    T t136 = (1.0/3.0)*t135;
    T t137 = Q_plus[3]*t106;
    T t138 = Q_plus[4]*t107;
    T t139 = -Q_plus[3]*t138 - Q_plus[5]*t108;
    T t140 = -Q_plus[2]*t137*t46 + n[1]*t139;
    T t141 = 2*t128 + t131;
    T t142 = t121 + t124;
    T t143 = n[1]*t97;
    T t144 = std::pow(Q_plus[5], 2);
    T t145 = n[0]*t111 + n[1]*(-std::pow(Q_plus[4], 2)*t107 - t109*t144 - t112);
    T t146 = n[0]*t120;
    T t147 = t127*t129;
    T t148 = (1.0/3.0)*t147;
    T t149 = 2*t121 + t124;
    T t150 = (1.0/3.0)*t128;
    T t151 = (2.0/3.0)*t131 + t150;
    T t152 = Q_plus[5]*t106;
    T t153 = -Q_plus[4]*t152*t62 + n[0]*t139;
    T t154 = t128 + 2*t131;
    T t155 = Q_plus[5]*Q_plus[7];
    T t156 = n[0]*(-Q_plus[6]*t108 - Q_plus[7]*t110) + n[1]*(-Q_plus[6]*t138 - t109*t155);
    T t157 = Q_plus[6]*t116;
    T t158 = t127*t157;
    T t159 = Q_plus[7]*t116;
    T t160 = t119*t159;
    T t161 = (1.0/3.0)*t160;
    T t162 = n[0]*t161;
    T t163 = t130*t157;
    T t164 = n[1]*t161;
    T t165 = t132 + t150;
    T t166 = (6.0/5.0)*Q_plus[6];
    T t167 = (4.0/5.0)*t137;
    T t168 = (4.0/5.0)*t106;
    T t169 = n[0]*(-Q_plus[2]*Qaux_plus[0]*t167 + (4.0/5.0)*Q_plus[3]*Q_plus[5]*Qaux_plus[1]*t101*t103*t105*t99 - Q_plus[4]*Qaux_plus[1]*t167 - Q_plus[7]*t108 - Q_plus[7]*t167 + (4.0/5.0)*Qaux_plus[0]*t101*t103*t105*t113*t99 - t137*t166) + n[1]*(-Q_plus[2]*Q_plus[5]*Qaux_plus[0]*t168 + (4.0/5.0)*Q_plus[3]*Q_plus[5]*Qaux_plus[0]*t101*t103*t105*t99 - Q_plus[4]*Q_plus[5]*Qaux_plus[1]*t168 - Q_plus[7]*t138 + (4.0/5.0)*Qaux_plus[1]*t101*t103*t105*t144*t99 - t152*t166 - t155*t168);
    T t170 = (2.0/5.0)*Qaux_plus[0];
    T t171 = t131*t170;
    T t172 = (2.0/5.0)*t119;
    T t173 = t126*t172;
    T t174 = Qaux_plus[0]*t173;
    T t175 = n[0]*(t160 + t174) + t171;
    T t176 = (4.0/5.0)*t119;
    T t177 = t129*t172;
    T t178 = Qaux_plus[1]*t177;
    T t179 = (2.0/5.0)*Qaux_plus[1];
    T t180 = -3.0/5.0*t119*t157 + t120*t179 + t123*t170 + (3.0/5.0)*t160;
    T t181 = n[0]*(-Qaux_plus[0]*t126*t176 - t178 + t180) - t171;
    T t182 = t128*t179;
    T t183 = n[1]*(t160 + t178) + t182;
    T t184 = n[1]*(-Qaux_plus[1]*t129*t176 - t174 + t180) - t182;
    T t185 = (3.0/5.0)*t128 + (3.0/5.0)*t131;
    T t186 = n[0]*(t123 + t173) + n[1]*(t120 + t177);
    T t187 = 0.5*std::max(std::fabs(eigenvalues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::max(std::fabs(eigenvalues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)), std::max(std::fabs(eigenvalues(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::max(std::fabs(eigenvalues(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)), std::max(std::fabs(eigenvalues(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::max(std::fabs(eigenvalues(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)), std::max(std::fabs(eigenvalues(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::max(std::fabs(eigenvalues(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)), std::max(std::fabs(eigenvalues(4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::max(std::fabs(eigenvalues(4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)), std::max(std::fabs(eigenvalues(5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::max(std::fabs(eigenvalues(5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)), std::max(std::fabs(eigenvalues(6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::max(std::fabs(eigenvalues(6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)), std::max(std::fabs(eigenvalues(7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t6, t22, t33, t41, t42, t44, 0, 0, 0, t49, t50, t51, t43, 0, 0, 0, t52, t54, t55, t57, t58, t60, 0, 0, 0, t63, t56, 0, t64, t51, 0, 0, 0, t66, t68, t72, t73, t74, t51, t75, 0, t79, t85, t91, t93, t94, t95, t96)), std::fabs(eigenvalues(7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t98, t114, t125, t133, t134, t136, 0, 0, 0, t140, t141, t142, t135, 0, 0, 0, t143, t145, t146, t148, t149, t151, 0, 0, 0, t153, t147, 0, t154, t142, 0, 0, 0, t156, t158, t162, t163, t164, t142, t165, 0, t169, t175, t181, t183, t184, t185, t186)))))))))))))))));
    T t188 = t187*(-t2 + t4);
    T t189 = (1.0/2.0)*p[0];
    T t190 = -std::pow(Q_plus[1], 2)*t189 + (1.0/2.0)*p[0]*t99;
    T t191 = t187*(Q_plus[2]*t115*t4 - t30);
    T t192 = Q_plus[3]*t115*t4 - t34;
    T t193 = t187*t192;
    T t194 = 0.27777777777777785*n[0];
    T t195 = 0.8872983346207417*t25;
    T t196 = 0.1127016653792583*t25;
    T t197 = t118*t196 + t195*t26;
    T t198 = t197*(0.1127016653792583*t122 + 0.8872983346207417*t30);
    T t199 = 0.5*t25;
    T t200 = t118*t199 + t199*t26;
    T t201 = 0.44444444444444442*t200;
    T t202 = t201*(0.5*t122 + 0.5*t30);
    T t203 = t118*t195 + t196*t26;
    T t204 = t203*(0.8872983346207417*t122 + 0.1127016653792583*t30);
    T t205 = 0.5*t192;
    T t206 = 0.27777777777777785*n[1];
    T t207 = Q_plus[5]*t115*t4 - t37;
    T t208 = 0.5*t207;
    T t209 = t205*(-n[0]*t202 - t194*t198 - t194*t204) + t208*(-n[1]*t202 - t198*t206 - t204*t206);
    T t210 = t187*(Q_plus[4]*t115*t4 - t24);
    T t211 = t187*t207;
    T t212 = t197*(0.1127016653792583*t117 + 0.8872983346207417*t24);
    T t213 = t201*(0.5*t117 + 0.5*t24);
    T t214 = t203*(0.8872983346207417*t117 + 0.1127016653792583*t24);
    T t215 = t205*(-n[0]*t213 - t194*t212 - t194*t214) + t208*(-n[1]*t213 - t206*t212 - t206*t214);
    T t216 = t187*(Q_plus[6]*t115*t4 - t67);
    T t217 = t187*(Q_plus[7]*t115*t4 - t69);
    T t218 = -6.0/5.0*t197*(0.1127016653792583*t157 + 0.8872983346207417*t67) + (1.0/5.0)*t197*(0.1127016653792583*t159 + 0.8872983346207417*t69);
    T t219 = -0.53333333333333333*t200*(0.5*t157 + 0.5*t67) + 0.088888888888888892*t200*(0.5*t159 + 0.5*t69);
    T t220 = -6.0/5.0*t203*(0.8872983346207417*t157 + 0.1127016653792583*t67) + (1.0/5.0)*t203*(0.8872983346207417*t159 + 0.1127016653792583*t69);
    T t221 = t205*(n[0]*t219 + t194*t218 + t194*t220) + t208*(n[1]*t219 + t206*t218 + t206*t220);
    T t222 = std::pow(Q_minus[1], 2)*t189 - t189*t7;
    SimpleArray<T, 16> res;
    res[0] = 0;
    res[1] = t188;
    res[2] = n[0]*t190 + t191;
    res[3] = t193 + t209;
    res[4] = n[1]*t190 + t210;
    res[5] = t211 + t215;
    res[6] = t216;
    res[7] = t217 + t221;
    res[8] = 0;
    res[9] = -t188;
    res[10] = n[0]*t222 - t191;
    res[11] = -t193 + t209;
    res[12] = n[1]*t222 - t210;
    res[13] = -t211 + t215;
    res[14] = -t216;
    res[15] = -t217 + t221;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = p[0]*Q[1];
    T t1 = n[0]*t0;
    T t2 = std::pow(Qaux[6], 2);
    T t3 = Q[2]*t2;
    T t4 = (1.0/3.0)*t2;
    T t5 = Q[3]*t4;
    T t6 = -Q[4]*t3 - Q[5]*t5;
    T t7 = -p[0]*Q[1];
    T t8 = std::pow(Q[3], 2);
    T t9 = n[0]*(-std::pow(Q[2], 2)*t2 - t4*t8 - t7) + n[1]*t6;
    T t10 = Qaux[6]*Q[4];
    T t11 = n[1]*t10;
    T t12 = Qaux[6]*Q[2];
    T t13 = n[0]*t12;
    T t14 = t11 + 2*t13;
    T t15 = Qaux[6]*n[0];
    T t16 = Q[3]*t15;
    T t17 = Qaux[6]*n[1];
    T t18 = Q[5]*t17;
    T t19 = (1.0/3.0)*t18;
    T t20 = (2.0/3.0)*t16 + t19;
    T t21 = n[1]*t12;
    T t22 = Q[3]*t17;
    T t23 = (1.0/3.0)*t22;
    T t24 = Q[5]*t3;
    T t25 = Q[4]*t2;
    T t26 = Q[3]*t25;
    T t27 = -t24 - t26;
    T t28 = -2*n[0]*Q[3]*t3 + n[1]*t27;
    T t29 = 2*t16 + t18;
    T t30 = t11 + t13;
    T t31 = n[1]*t0;
    T t32 = std::pow(Q[5], 2);
    T t33 = n[0]*t6 + n[1]*(-std::pow(Q[4], 2)*t2 - t32*t4 - t7);
    T t34 = n[0]*t10;
    T t35 = Q[5]*t15;
    T t36 = (1.0/3.0)*t35;
    T t37 = 2*t11 + t13;
    T t38 = (1.0/3.0)*t16;
    T t39 = (2.0/3.0)*t18 + t38;
    T t40 = Q[5]*t25;
    T t41 = n[0]*t27 - 2*n[1]*t40;
    T t42 = t16 + 2*t18;
    T t43 = Q[5]*Q[7];
    T t44 = n[0]*(-Q[6]*t3 - Q[7]*t5) + n[1]*(-Q[6]*t25 - t4*t43);
    T t45 = Q[6]*t15;
    T t46 = Qaux[6]*Q[7];
    T t47 = (1.0/3.0)*t46;
    T t48 = n[0]*t47;
    T t49 = Q[6]*t17;
    T t50 = n[1]*t47;
    T t51 = t19 + t38;
    T t52 = (3.0/5.0)*Q[6];
    T t53 = Q[3]*t2;
    T t54 = Qaux[0]*Q[3];
    T t55 = (2.0/5.0)*Qaux[1];
    T t56 = (2.0/5.0)*Qaux[0];
    T t57 = n[0]*((2.0/5.0)*Qaux[0]*t2*t8 + (2.0/5.0)*Qaux[1]*Q[3]*Q[5]*t2 - Q[7]*t3 - 2.0/5.0*Q[7]*t53 - t26*t55 - 2.0/5.0*t3*t54 - t52*t53) + n[1]*((2.0/5.0)*Qaux[0]*Q[3]*Q[5]*t2 + (2.0/5.0)*Qaux[1]*t2*t32 - Q[5]*t2*t52 - Q[7]*t25 - 2.0/5.0*t2*t43 - t24*t56 - t40*t55);
    T t58 = t18*t56;
    T t59 = (2.0/5.0)*Qaux[6];
    T t60 = Q[3]*t59;
    T t61 = Qaux[0]*t60;
    T t62 = n[0]*(t46 + t61) + t58;
    T t63 = (4.0/5.0)*Qaux[6];
    T t64 = Q[5]*t59;
    T t65 = Qaux[1]*t64;
    T t66 = -Qaux[6]*t52 + t10*t55 + t12*t56 + (3.0/5.0)*t46;
    T t67 = n[0]*(-t54*t63 - t65 + t66) - t58;
    T t68 = t16*t55;
    T t69 = n[1]*(t46 + t65) + t68;
    T t70 = n[1]*(-Qaux[1]*Q[5]*t63 - t61 + t66) - t68;
    T t71 = (3.0/5.0)*t16 + (3.0/5.0)*t18;
    T t72 = n[0]*(t12 + t60) + n[1]*(t10 + t64);
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(eigenvalues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)), std::max(std::fabs(eigenvalues(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)), std::max(std::fabs(eigenvalues(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)), std::max(std::fabs(eigenvalues(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)), std::max(std::fabs(eigenvalues(4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)), std::max(std::fabs(eigenvalues(5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)), std::max(std::fabs(eigenvalues(6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)), std::fabs(eigenvalues(7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n[0], 0, n[1], 0, 0, 0, t1, t9, t14, t20, t21, t23, 0, 0, 0, t28, t29, t30, t22, 0, 0, 0, t31, t33, t34, t36, t37, t39, 0, 0, 0, t41, t35, 0, t42, t30, 0, 0, 0, t44, t45, t48, t49, t50, t30, t51, 0, t57, t62, t67, t69, t70, t71, t72)))))))));
    return res;
    }

};
