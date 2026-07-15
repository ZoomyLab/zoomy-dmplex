#include <cstdio>
#include <cmath>
#include "UserFunctions.H"

int main() {
    int fail = 0;
    // ---- eigensystem: SWE quasilinear A (n=2), u=2, c=sqrt(g h)=3
    //   A = [[0,1],[c^2-u^2, 2u]]  -> eigenvalues u-c=-1, u+c=5
    const double u = 2.0, c = 3.0;
    double a[4] = {0.0, 1.0, c*c - u*u, 2*u};
    double lam[2], R[4], L[4];
    for (int i = 0; i < 2; ++i) lam[i] = eigensystem(i, a[0], a[1], a[2], a[3]);
    for (int i = 0; i < 4; ++i) R[i] = eigensystem(2 + i, a[0], a[1], a[2], a[3]);
    for (int i = 0; i < 4; ++i) L[i] = eigensystem(6 + i, a[0], a[1], a[2], a[3]);
    printf("eig lambda = %.6f %.6f   (expect -1, 5 in some order)\n", lam[0], lam[1]);
    double lo = std::min(lam[0], lam[1]), hi = std::max(lam[0], lam[1]);
    if (std::fabs(lo - (u - c)) > 1e-10 || std::fabs(hi - (u + c)) > 1e-10) { printf("  FAIL eigenvalues\n"); fail++; }

    // reconstruct A = R * diag(lam) * L  -> must recover the input exactly
    double err = 0.0;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        double s = 0.0;
        for (int k = 0; k < 2; ++k) s += R[i*2+k] * lam[k] * L[k*2+j];
        err = std::max(err, std::fabs(s - a[i*2+j]));
    }
    printf("||R diag(lam) L - A||_inf = %.3e  (consistency of the SAME eigenbasis)\n", err);
    if (err > 1e-10) { printf("  FAIL reconstruction\n"); fail++; }

    // L must be the inverse of R
    double ierr = 0.0;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        double s = 0.0;
        for (int k = 0; k < 2; ++k) s += L[i*2+k] * R[k*2+j];
        ierr = std::max(ierr, std::fabs(s - (i == j ? 1.0 : 0.0)));
    }
    printf("||L R - I||_inf = %.3e\n", ierr);
    if (ierr > 1e-10) { printf("  FAIL L != R^-1\n"); fail++; }

    // Roe dissipation |A| = R |Lam| L must be well defined & symmetric-consistent
    double absA[4];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        double s = 0.0;
        for (int k = 0; k < 2; ++k) s += R[i*2+k] * std::fabs(lam[k]) * L[k*2+j];
        absA[i*2+j] = s;
    }
    printf("|A| = [[%.4f, %.4f], [%.4f, %.4f]]\n", absA[0], absA[1], absA[2], absA[3]);

    // ---- eigenvalues: the lambda-only kernel must agree with eigensystem's lambda
    double ev[2];
    for (int i = 0; i < 2; ++i) ev[i] = eigenvalues(i, a[0], a[1], a[2], a[3]);
    printf("eigenvalues     = %.6f %.6f   (expect -1, 5)\n", ev[0], ev[1]);
    double evlo = std::min(ev[0], ev[1]), evhi = std::max(ev[0], ev[1]);
    if (std::fabs(evlo - (u - c)) > 1e-10 || std::fabs(evhi - (u + c)) > 1e-10) { printf("  FAIL eigenvalues\n"); fail++; }
    // must match eigensystem's spectrum as a SET (order is not guaranteed)
    if (std::fabs(evlo - lo) > 1e-10 || std::fabs(evhi - hi) > 1e-10) { printf("  FAIL eigenvalues != eigensystem spectrum\n"); fail++; }
    // max|lambda| is what local_max_abs_eigenvalue actually consumes
    double mx = std::max(std::fabs(ev[0]), std::fabs(ev[1]));
    printf("max|lambda|     = %.6f   (expect 5 = |u|+c)\n", mx);
    if (std::fabs(mx - 5.0) > 1e-10) { printf("  FAIL max|lambda|\n"); fail++; }
    // cache invalidation on a different matrix
    double e2 = eigenvalues(0, 2.0, 0.0, 0.0, 3.0);
    double e3 = eigenvalues(1, 2.0, 0.0, 0.0, 3.0);
    printf("eigenvalues(diag(2,3)) = %.6f %.6f   (expect 2, 3 -> cache invalidates)\n", e2, e3);
    if (std::fabs(std::min(e2,e3) - 2.0) > 1e-10 || std::fabs(std::max(e2,e3) - 3.0) > 1e-10) { printf("  FAIL eigenvalues cache\n"); fail++; }

    // ---- solve: A x = b, n=3, known answer
    //   A = [[2,1,0],[1,3,1],[0,1,4]], b = [3,9,13] -> x = [1,1,3]
    double x0 = solve(0, 2.,1.,0., 1.,3.,1., 0.,1.,4., 3.,7.,13.);
    double x1 = solve(1, 2.,1.,0., 1.,3.,1., 0.,1.,4., 3.,7.,13.);
    double x2 = solve(2, 2.,1.,0., 1.,3.,1., 0.,1.,4., 3.,7.,13.);
    printf("solve x = %.10f %.10f %.10f   (expect 1, 1, 3)\n", x0, x1, x2);
    if (std::fabs(x0-1) > 1e-10 || std::fabs(x1-1) > 1e-10 || std::fabs(x2-3) > 1e-10) { printf("  FAIL solve\n"); fail++; }

    // ---- cache invalidation: a DIFFERENT matrix must not return the stale answer
    double y0 = solve(0, 1.,0.,0., 0.,1.,0., 0.,0.,1., 7.,8.,9.);
    double y1 = solve(1, 1.,0.,0., 0.,1.,0., 0.,0.,1., 7.,8.,9.);
    printf("solve(I, [7,8,9]) = %.6f %.6f   (expect 7, 8 -> cache invalidates)\n", y0, y1);
    if (std::fabs(y0-7) > 1e-10 || std::fabs(y1-8) > 1e-10) { printf("  FAIL cache invalidation\n"); fail++; }

    printf(fail ? "\nRESULT: %d FAILURES\n" : "\nRESULT: all pass\n", fail);
    return fail;
}
