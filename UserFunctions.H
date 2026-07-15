#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <petscdmplex.h>

// ── REQ-168 SPEC ADDENDUM (speed, 2026-07-15) ────────────────────────────
// (1) The eig kernels MUST be inf-tolerant: at order 2 the a-posteriori MOOD
//     path evaluates the wave speed on NOT-YET-CORRECTED candidate states,
//     which are non-finite at a dry front BY DESIGN. Contract: report +inf for
//     a non-finite face — an infinite wave speed is exactly right for a garbage
//     candidate (dt clamps, MOOD flags and drops the cell to order 1).
//
//     ⚠ dmplex needs this MORE than numpy does, and for the OPPOSITE reason.
//     numpy/LAPACK RAISE on non-finite input ("LinAlgError: Array must not
//     contain infs") — loud. MEASURED here: Eigen does NOT raise and does NOT
//     hang; EigenSolver::info() returns NoConvergence and the eigenvalues are
//     left GARBAGE. Unguarded, dmplex would SILENTLY consume a wrong wave speed
//     exactly where numpy crashes. So both kernels below pre-check their inputs
//     AND check info() (which also catches genuine non-convergence on finite
//     input). Never trust an unchecked Eigen decomposition.
//
// (2) The wave-speed path must NEVER invert the eigenbasis (a dry-cell Jacobian
//     is defective, so inv(V) fails even when the eigenvalues are fine).
//     dmplex satisfies this BY CONSTRUCTION: `eigenvalues` is a SEPARATE kernel
//     that never forms R or L. Only `eigensystem` — which just Roe requests —
//     inverts, and may legitimately fail on a defective matrix. dmplex needs no
//     lazy idx<n split; the split is the two kernels.
#define ZOOMY_EIG_INF (std::numeric_limits<double>::infinity())

// Backend implementations of the opaque placeholder kernels the zoomy_core C++
// printer emits (REQ-38/REQ-65). The generated Model.H calls these by bare name,
// so this header must be included BEFORE Model.H in every translation unit that
// instantiates the kernels (unqualified lookup at the template definition point
// then resolves them).

// ── ZoomyMesh ────────────────────────────────────────────────────────────
// Mesh handle the field-level, mesh-aware update_aux_variables(Q, Qaux, dt,
// mesh) overload (REQ-65) passes to compute_derivative. It carries the DMPlex
// the LSQ/Green-Gauss stencil reads from. Per-field cell data (Q[i] / Qaux[k])
// are plain contiguous arrays indexed by (cell point - cStart); the solver
// de-interleaves the DMPlex Vec into them before the call (see
// TransportStep::RefreshDerivativeAux).
struct ZoomyMesh {
    DM dm;            // DMPlex (cell-centred FV); geometry via DMPlexGetGeometryFVM
    PetscInt cStart;  // first cell point (field arrays are indexed c - cStart)
    PetscInt cEnd;
};

// ── compute_derivative ───────────────────────────────────────────────────
// result[c] = D^(dx,dy,dz) source over the mesh (one scalar field). Green-Gauss
// cell gradient: grad = (1/V) Σ_faces face_value · n_f, then pick the requested
// axis. Total order > 1 is not supported (the SME aux_registry only needs
// first derivatives); higher orders would need a wider LSQ stencil.
// Boundary faces use a zero-Neumann face value (the interior cell value), which
// is consistent for the smooth bathymetry/depth gradients consumed downstream.
// Serial (no halo exchange); parallel needs a ghosted source — see REQ note.
template <typename T>
inline void compute_derivative(T* result, const T* source,
                               int dx, int dy, int dz, const ZoomyMesh& mesh) {
    const int total = dx + dy + dz;
    const int axis = (dx == 1 ? 0 : (dy == 1 ? 1 : 2));
    const PetscInt cStart = mesh.cStart, cEnd = mesh.cEnd;
    DM dm = mesh.dm;
    PetscInt dim; DMGetDimension(dm, &dim);   // mesh dimension (no Model dependency)

    for (PetscInt c = cStart; c < cEnd; ++c) result[c - cStart] = (T)0;
    if (total == 0) {                       // D^(0,0,0) = identity
        for (PetscInt c = cStart; c < cEnd; ++c) result[c - cStart] = source[c - cStart];
        return;
    }
    Vec faceGeom, cellGeom;
    if (DMPlexGetGeometryFVM(dm, &faceGeom, &cellGeom, NULL)) return;
    const PetscScalar *fG, *cG;
    VecGetArrayRead(faceGeom, &fG); VecGetArrayRead(cellGeom, &cG);
    DM dmFace; VecGetDM(faceGeom, &dmFace);
    PetscSection secFace; DMGetLocalSection(dmFace, &secFace);
    DM dmCell; VecGetDM(cellGeom, &dmCell);
    PetscSection secCell; DMGetLocalSection(dmCell, &secCell);

    PetscInt fStart, fEnd; DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);
    std::vector<T> grad((cEnd - cStart) * dim, (T)0);

    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscInt off; PetscSectionGetOffset(secFace, f, &off);
        const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fG[off];
        const PetscInt *cells; PetscInt nc;
        DMPlexGetSupportSize(dm, f, &nc); DMPlexGetSupport(dm, f, &cells);
        if (nc == 2) {
            T sL = source[cells[0] - cStart], sR = source[cells[1] - cStart];
            T fv = (T)0.5 * (sL + sR);
            for (int d = 0; d < dim; ++d) {
                T v = fv * (T)fg->normal[d];
                grad[(cells[0] - cStart) * dim + d] += v;
                grad[(cells[1] - cStart) * dim + d] -= v;
            }
        } else if (nc == 1) {               // boundary: zero-Neumann (face = cell)
            T fv = source[cells[0] - cStart];
            for (int d = 0; d < dim; ++d)
                grad[(cells[0] - cStart) * dim + d] += fv * (T)fg->normal[d];
        }
    }
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscInt offc; PetscSectionGetOffset(secCell, c, &offc);
        const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cG[offc];
        T vol = (T)cg->volume;
        if (vol > (T)0) result[c - cStart] = grad[(c - cStart) * dim + axis] / vol;
    }
    VecRestoreArrayRead(faceGeom, &fG); VecRestoreArrayRead(cellGeom, &cG);
}

// ── eigensystem ──────────────────────────────────────────────────────────
// REQ-168 (user directive 2026-07-15): every backend implements every opaque
// UserFunction. Mirrors foam's `numerics::eigensystem` and numpy's
// `_eigensystem_numpy`: given the component index `idx` followed by the flat
// row-major normal-projected quasilinear matrix A (n*n entries), return the
// idx-th entry of the stacked decomposition
//     [ eigenvalues lambda (n), right eigenvectors R (n*n), left L = R^-1 (n*n) ]
// The printer emits the n+2n^2 component reads of ONE face with identical
// A_flat args; the 1-slot cache makes them share a single decomposition —
// required for CORRECTNESS (R, |Lambda| and L must come from the SAME
// eigenbasis so the Roe dissipation |A| = R|Lambda|L is consistent), not just
// speed. |A| is invariant to eigenpair ordering, so Eigen-vs-LAPACK order is
// harmless. Realised with Eigen (PETSc is configured --download-eigen).
template <typename I, typename... Ts>
inline double eigensystem(I idx_, Ts... a_flat_args) {
    const int idx = static_cast<int>(idx_);
    const double a_flat[] = { static_cast<double>(a_flat_args)... };
    constexpr int M = sizeof...(a_flat_args);
    const int n = static_cast<int>(std::lround(std::sqrt(double(M))));

    static thread_local std::vector<double> cache_key, cache_out;
    bool hit = (static_cast<int>(cache_key.size()) == M);
    for (int k = 0; hit && k < M; ++k) hit = (cache_key[k] == a_flat[k]);

    if (!hit) {
        // REQ-168 addendum (1): non-finite candidate (MOOD, by design) or a
        // decomposition Eigen could not converge -> lambda = +inf, R = L = I.
        // Identity (not zero) for R/L so a Roe |A| = R|Lambda|L collapses to
        // inf*I -- an infinite dissipation the dt clamp / MOOD flag can act on
        // -- rather than 0*inf = NaN, which would poison the whole flux row.
        bool finite = true;
        for (int k = 0; k < M && finite; ++k) finite = std::isfinite(a_flat[k]);
        Eigen::MatrixXd A(n, n);
        if (finite)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) A(i, j) = a_flat[i * n + j];
        Eigen::EigenSolver<Eigen::MatrixXd> es;
        if (finite) es.compute(A);
        const bool ok = finite && (es.info() == Eigen::Success);

        cache_out.assign(n + 2 * n * n, 0.0);
        if (!ok) {
            for (int i = 0; i < n; ++i) cache_out[i] = ZOOMY_EIG_INF;
            for (int i = 0; i < n; ++i) {
                cache_out[n + i * n + i] = 1.0;                  // R = I
                cache_out[n + n * n + i * n + i] = 1.0;          // L = I
            }
        } else {
            const Eigen::VectorXd w = es.eigenvalues().real();
            const Eigen::MatrixXd V = es.eigenvectors().real();  // cols = eigvecs
            Eigen::MatrixXd L;
            Eigen::FullPivLU<Eigen::MatrixXd> lu(V);
            if (lu.isInvertible()) L = lu.inverse();
            else                   L = V.completeOrthogonalDecomposition().pseudoInverse();
            for (int i = 0; i < n; ++i) cache_out[i] = w(i);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) cache_out[n + i * n + j] = V(i, j);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) cache_out[n + n * n + i * n + j] = L(i, j);
        }
        cache_key.assign(a_flat, a_flat + M);
    }
    return cache_out[idx];
}

// ── eigenvalues ──────────────────────────────────────────────────────────
// The λ-only companion of `eigensystem` (core kernel_functions.eigenvalues):
// idx-th eigenvalue (real part) of the row-major n*n A_flat, n = sqrt(M).
// `local_max_abs_eigenvalue` needs only max|λ| for the CFL bound, so this skips
// the eigenvector solve AND the R⁻¹ that `eigensystem` computes — ~2/3 less work
// per face per step, and it drops the inverse, which is the part that can fail
// near wet/dry (a degenerate spectrum makes V singular). This is the kernel
// dmplex/jax/amrex each asked core for from a different angle (REQ-168 GAP 1).
//
// Unlike `eigensystem`, the 1-slot cache here is a pure OPTIMISATION, not a
// correctness requirement: the printer emits the n component reads of one face
// with an identical A, and max|λ| is invariant to eigenpair ORDER, so even a
// per-call re-decomposition with a different order would give the same answer.
// (For `eigensystem` the shared basis IS correctness — R, |Λ| and L must agree.)
template <typename I, typename... Ts>
inline double eigenvalues(I idx_, Ts... a_flat_args) {
    const int idx = static_cast<int>(idx_);
    const double a_flat[] = { static_cast<double>(a_flat_args)... };
    constexpr int M = sizeof...(a_flat_args);
    const int n = static_cast<int>(std::lround(std::sqrt(double(M))));

    static thread_local std::vector<double> cache_key, cache_out;
    bool hit = (static_cast<int>(cache_key.size()) == M);
    for (int k = 0; hit && k < M; ++k) hit = (cache_key[k] == a_flat[k]);

    if (!hit) {
        // REQ-168 addendum (1): MOOD evaluates the wave speed on uncorrected
        // candidates that are non-finite at a dry front BY DESIGN. Report +inf
        // (dt clamps, MOOD flags). Eigen returns NoConvergence + GARBAGE rather
        // than raising, so info() must be checked or the garbage is consumed
        // silently — see the header note.
        bool finite = true;
        for (int k = 0; k < M && finite; ++k) finite = std::isfinite(a_flat[k]);
        Eigen::MatrixXd A(n, n);
        if (finite)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) A(i, j) = a_flat[i * n + j];
        Eigen::EigenSolver<Eigen::MatrixXd> es;
        if (finite) es.compute(A, /*computeEigenvectors=*/false);

        cache_out.assign(n, ZOOMY_EIG_INF);
        if (finite && es.info() == Eigen::Success) {
            const Eigen::VectorXd w = es.eigenvalues().real();
            cache_out.assign(w.data(), w.data() + n);
        }
        cache_key.assign(a_flat, a_flat + M);
    }
    return cache_out[idx];
}

// ── solve ────────────────────────────────────────────────────────────────
// The idx-th component of the PER-CELL dense linear solve A^-1 b, emitted by
// the NSM's point-implicit source linearization (`source_treatment='linearized'`:
// S_lin = (I - dt*J)^-1 S, so A = I - dt*J, b = S). Args are `idx` then the flat
// row-major A (n*n) followed by b (n) => M = n*n + n. LOCAL per cell (no mesh
// coupling) — unrelated to the VAM pressure solve, which is a global PETSc KSP.
// Same 1-slot cache: the n component reads of one cell share one factorisation.
template <typename I, typename... Ts>
inline double solve(I idx_, Ts... args) {
    const int idx = static_cast<int>(idx_);
    const double v[] = { static_cast<double>(args)... };
    constexpr int M = sizeof...(args);
    // M = n^2 + n  =>  n = (-1 + sqrt(1+4M))/2
    const int n = static_cast<int>(std::lround((-1.0 + std::sqrt(1.0 + 4.0 * double(M))) / 2.0));

    static thread_local std::vector<double> cache_key, cache_out;
    bool hit = (static_cast<int>(cache_key.size()) == M);
    for (int k = 0; hit && k < M; ++k) hit = (cache_key[k] == v[k]);

    if (!hit) {
        Eigen::MatrixXd A(n, n); Eigen::VectorXd b(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) A(i, j) = v[i * n + j];
        for (int i = 0; i < n; ++i) b(i) = v[n * n + i];
        const Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
        cache_out.assign(x.data(), x.data() + n);
        cache_key.assign(v, v + M);
    }
    return cache_out[idx];
}
