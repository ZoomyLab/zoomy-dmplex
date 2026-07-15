#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <petscdmplex.h>

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
        Eigen::MatrixXd A(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) A(i, j) = a_flat[i * n + j];
        Eigen::EigenSolver<Eigen::MatrixXd> es(A);
        const Eigen::VectorXd w = es.eigenvalues().real();
        const Eigen::MatrixXd V = es.eigenvectors().real();     // cols = eigvecs
        Eigen::MatrixXd L;
        Eigen::FullPivLU<Eigen::MatrixXd> lu(V);
        if (lu.isInvertible()) L = lu.inverse();
        else                   L = V.completeOrthogonalDecomposition().pseudoInverse();

        cache_out.assign(n + 2 * n * n, 0.0);
        for (int i = 0; i < n; ++i) cache_out[i] = w(i);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) cache_out[n + i * n + j] = V(i, j);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) cache_out[n + n * n + i * n + j] = L(i, j);
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
