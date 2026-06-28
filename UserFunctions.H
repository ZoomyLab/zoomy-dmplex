#pragma once
#include <cmath>
#include <vector>
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
