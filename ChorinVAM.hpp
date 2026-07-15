#ifndef CHORIN_VAM_HPP
#define CHORIN_VAM_HPP

#include "MUSCLSolver.hpp"      // pulls Model.H -> defines ZOOMY_MODEL_IS_VAM iff VAM
#include <petscksp.h>

// ─────────────────────────────────────────────────────────────────────────
// REQ-169 (speed): main.cpp includes this header unconditionally and does
// `std::make_unique<ChorinVAMSolver>()`, so the class must be COMPLETE in every
// build. ChorinVAMSolver is not a template, so a static_assert in its body is
// evaluated at PARSE — with a non-VAM Model.H (e.g. SWE, n_dof_q=4) against the
// checked-in VAM ChorinOps.H (VAM_NS=8) it fired and the whole tree stopped
// compiling. The assert is right (it catches mismatched generated headers
// reading wrong slots); it just must not fire for a model that never uses it.
//
// So the real body compiles only for a VAM tree. The selector is emitted BY
// create_vam_model.py INTO Model.H — the same generated file that defines
// n_dof_q — so it is rewritten whenever n_dof_q is and cannot go stale against
// it. (A -D build flag would be a second source of truth that can disagree with
// the headers in either direction: set for an SWE tree -> the assert fires
// again; forgotten for a VAM tree -> the stub silently replaces the solver.)
// ─────────────────────────────────────────────────────────────────────────
#if !defined(ZOOMY_MODEL_IS_VAM)

class ChorinVAMSolver : public MUSCLSolver {
public:
    PetscErrorCode Run(int, char **) override {
        PetscFunctionBeginUser;
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,
                "solver.method=\"chorin\" but this binary was generated for a "
                "non-VAM model (Model.H carries no ZOOMY_MODEL_IS_VAM). Re-run "
                "create_vam_model.py --dimension {2|3} and rebuild "
                "(make clean_all && make CPU).");
    }
};

#else   // ── VAM tree: the real solver ─────────────────────────────────────
#include "ChorinOps.H"

// ─────────────────────────────────────────────────────────────────────────
// VAM Chorin pressure-split solver (hyperbolic-elliptic).
//
//   predictor (pressure-free FV) -> pressure (global elliptic KSP, matrix-free
//   on the AFFINE operator) -> corrector (pointwise, uses grad P).
//
// The predictor NSM DROPS the pressure modes (no evolution eq), so the evolved
// state X is NS dof: dim=2 -> [b,h,q_0,q_1,r_0,r_1]; dim=3 -> [b,h,q_x0,q_x1,
// q_y0,q_y1,r0,r1]. The pressure P (NP) lives in a SEPARATE vector Pv. The
// generated ChorinOps read the FULL NF-state Q = [X(NS), P(NP)] — assembled per
// cell here. The corrector updates the VAM_CORR_IDX slots. Derivative-aux array
// CA (NCA/cell): the leading 2*NP*dim entries are the P first+second derivs
// (recomputed each matvec), the rest are frozen b/h(1st+2nd)+q(1st) (once/step).
//
// ⚠ ALL SHAPE CONSTANTS COME FROM THE GENERATED ChorinOps.H — never hard-code
// them. They change with `dimension` (dim=2: NF=8/NCA=10; dim=3: NF=10/NCA=20),
// and a stale hard-coded dim=3 layout indexes a dim=2 run silently out of range.
// ─────────────────────────────────────────────────────────────────────────
class ChorinVAMSolver : public MUSCLSolver {
    static constexpr int NS = VAM_NS;    // evolved state dof (== Model<Real>::n_dof_q)
    static constexpr int NP = VAM_NP;    // pressure modes
    static constexpr int NF = VAM_NF;    // full VAM state for ChorinOps
    static constexpr int NCA = VAM_NCA;  // derivative-aux entries per cell
    // the generated predictor MUST agree with the generated ops, or the
    // per-cell assembly below reads the wrong slots.
    static_assert(NS == Model<double>::n_dof_q,
                  "ChorinOps.H VAM_NS != Model.H n_dof_q — Model.H/ChorinOps.H "
                  "generated from different `dimension`; re-run create_vam_model.py");
    DM dmP = nullptr, dmCA = nullptr;
    Vec Pv = nullptr, Rhs = nullptr, R0 = nullptr, CA = nullptr, Pcur = nullptr;
    Mat Aop = nullptr;
    // REQ-172: assembled block Pmat (preconditioner operand) + the scalar
    // derivative stencils its blocks multiply. Sblk[i] == nullptr means block i
    // is the undifferentiated A0 (stencil = identity, no matrix needed).
    Mat Pmat = nullptr;
    Mat Sblk[VAM_N_PBLOCK] = {nullptr};
    bool use_gamg = true;                   // ZOOMY_VAM_PCNONE=1 restores PCNONE
    bool pmat_verified = false;             // consistency gate runs once, after step 1
    KSP ksp = nullptr;
    PetscInt cS = 0, cE = 0;
    PetscReal dt_chorin = 0.0;

public:
    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(VirtualSolver::Initialize(argc, argv));
        SetReconstruction(settings.solver.reconstruction_order >= 2 ? LINEAR : PCM);
        SetLimiters(settings.solver.limiter != "none");
        PetscCall(InitializeComponents());
        std::vector<std::string> names;
        for (int i = 0; i < Model<Real>::n_dof_q; ++i) names.push_back("q" + std::to_string(i));
        PetscCall(io->Setup3D(dmQ, Model<Real>::n_dof_q, names));
        PetscCall(this->SetupInitialConditions());
        PetscCall(EnforcePhysicalConstraints(X));
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cS, &cE));
        PetscCall(SetupChorin());

        PetscReal time = 0.0; PetscInt step = 0; (void)step;
        PetscCall(WriteSnapshot(time));
        while (time < settings.solver.t_end) {
            dt_chorin = std::max(ComputeTimeStep(), settings.solver.min_dt);
            // land exactly on t_end (see MUSCLSolver; REQ-159)
            if (time + dt_chorin > settings.solver.t_end) dt_chorin = settings.solver.t_end - time;
            PetscCall(Predictor(dt_chorin));
            PetscCall(ComputeFrozenAux());
            // run the Pmat consistency gate ONCE, here -- dt and the frozen
            // state are real only after the first predictor (see SetupChorin).
            if (use_gamg && !pmat_verified && getenv("ZOOMY_VAM_VERIFY_PMAT")) {
                PetscCall(AssemblePmat());
                PetscCall(VerifyPmat());
                pmat_verified = true;
            }
            if (getenv("VAM_NO_PRESSURE") == nullptr) {
                PetscCall(PressureSolve());
                PetscCall(Corrector());
            }
            PetscCall(EnforcePhysicalConstraints(X));
            time += dt_chorin; step++;
            if (rank == 0 && step % 10 == 0)
                std::cout << "Step " << step << " Time " << time << " dt " << dt_chorin << std::endl;
            PetscCall(WriteSnapshot(time));
        }
        if (rank == 0) std::cout << "[INFO] Finished (Chorin VAM)." << std::endl;
        PetscCall(TeardownChorin());
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    // Write a VTK snapshot without a TS (Monitor needs ts for its step-print).
    PetscErrorCode WriteSnapshot(PetscReal time) {
        PetscFunctionBeginUser;
        if (io->ShouldWrite(time)) {
            PetscCall(PackState(X, A, X_out));
            PetscCall(io->WriteVTK(dmOut, X_out, time));
            io->AdvanceSnapshot();
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Mirror SetupArchitecture's dmAux: clone the base mesh + a PetscFV with
    // `ndof` components (sets up the global section + point SF for
    // DMPlexPointGlobalRead/Ref — a hand-rolled section segfaults).
    PetscErrorCode CloneCellDM(PetscInt ndof, DM *out) {
        PetscFunctionBeginUser;
        DM d; PetscCall(DMClone(dmMesh, &d));
        PetscFV fv; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fv));
        PetscCall(PetscFVSetNumComponents(fv, ndof));
        PetscCall(PetscFVSetSpatialDimension(fv, Model<Real>::dimension));
        PetscCall(PetscFVSetType(fv, PETSCFVUPWIND));
        PetscCall(DMAddField(d, NULL, (PetscObject)fv));
        PetscCall(DMCreateDS(d));
        PetscCall(PetscFVDestroy(&fv));
        *out = d;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // ── REQ-172: assemble the pressure operator for a preconditioner ──────
    // Build the SCALAR Green-Gauss first-derivative matrix for `axis`, mirroring
    // UserFunctions.H::compute_derivative EXACTLY (same face loop, same 0.5
    // face averaging, same zero-Neumann boundary, same /vol). If these two ever
    // drift apart the Pmat stops being a preconditioner for the real operator --
    // which the consistency gate (VerifyPmat) is there to catch.
    //   interior face (L,R):  D[L][L] += 0.5*N,  D[L][R] += 0.5*N
    //                         D[R][L] -= 0.5*N,  D[R][R] -= 0.5*N
    //   boundary face (L):    D[L][L] += N          (face value = cell value)
    //   then row c /= vol[c]
    PetscErrorCode BuildDeriv1Mat(int axis, Mat *out) {
        PetscFunctionBeginUser;
        const PetscInt nc = cE - cS;
        Mat D;
        PetscCall(MatCreate(PETSC_COMM_WORLD, &D));
        PetscCall(MatSetSizes(D, nc, nc, PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCall(MatSetType(D, MATAIJ));
        PetscCall(MatSeqAIJSetPreallocation(D, 8, NULL));
        PetscCall(MatMPIAIJSetPreallocation(D, 8, NULL, 8, NULL));
        PetscCall(MatSetOption(D, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

        Vec faceGeom, cellGeom;
        PetscCall(DMPlexGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL));
        const PetscScalar *fG, *cG;
        PetscCall(VecGetArrayRead(faceGeom, &fG));
        PetscCall(VecGetArrayRead(cellGeom, &cG));
        DM dmFace; PetscCall(VecGetDM(faceGeom, &dmFace));
        PetscSection secFace; PetscCall(DMGetLocalSection(dmFace, &secFace));
        DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell));
        PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));

        PetscInt fS, fE; PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fS, &fE));
        for (PetscInt f = fS; f < fE; ++f) {
            PetscInt off; PetscCall(PetscSectionGetOffset(secFace, f, &off));
            const PetscFVFaceGeom *fg = (const PetscFVFaceGeom *)&fG[off];
            const PetscInt *cells; PetscInt ns;
            PetscCall(DMPlexGetSupportSize(dmQ, f, &ns));
            PetscCall(DMPlexGetSupport(dmQ, f, &cells));
            const PetscReal Nax = fg->normal[axis];         // PETSc normal is AREA-weighted
            if (ns == 2) {
                const PetscInt L = cells[0] - cS, R = cells[1] - cS;
                if (L < 0 || L >= nc || R < 0 || R >= nc) continue;
                PetscScalar h = 0.5 * Nax;
                PetscCall(MatSetValue(D, L, L,  h, ADD_VALUES));
                PetscCall(MatSetValue(D, L, R,  h, ADD_VALUES));
                PetscCall(MatSetValue(D, R, L, -h, ADD_VALUES));
                PetscCall(MatSetValue(D, R, R, -h, ADD_VALUES));
            } else if (ns == 1) {
                const PetscInt L = cells[0] - cS;
                if (L < 0 || L >= nc) continue;
                PetscCall(MatSetValue(D, L, L, Nax, ADD_VALUES));
            }
        }
        PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));
        // row c /= vol[c]
        Vec invvol; PetscCall(MatCreateVecs(D, NULL, &invvol));
        PetscScalar *iv; PetscCall(VecGetArray(invvol, &iv));
        for (PetscInt c = cS; c < cE; ++c) {
            PetscInt offc; PetscCall(PetscSectionGetOffset(secCell, c, &offc));
            const PetscFVCellGeom *cg = (const PetscFVCellGeom *)&cG[offc];
            iv[c - cS] = (cg->volume > 0.0) ? 1.0 / cg->volume : 0.0;
        }
        PetscCall(VecRestoreArray(invvol, &iv));
        PetscCall(MatDiagonalScale(D, invvol, NULL));       // left-scale rows
        PetscCall(VecDestroy(&invvol));
        PetscCall(VecRestoreArrayRead(faceGeom, &fG));
        PetscCall(VecRestoreArrayRead(cellGeom, &cG));
        *out = D;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Assemble Pmat = sum_blk B_blk(cell) (x) S_blk, where S_blk is the scalar
    // stencil the block multiplies (I for A0, Dx for Ax, Dx*Dx for Axx, ...) and
    // B_blk(cell) is the NP x NP coefficient block from the generated
    // vam_pressure_blocks() at that cell's frozen state.
    //   Pmat[(c,k), (c',l)] += B_blk[c][k][l] * S_blk[c][c']
    // ⚠ BLOCK SIZE NP IS THE WHOLE POINT (@amrex REQ-172 LFA: rho(D^-1 M) =
    // 3.9/1.8 > 1 => the mode block is coupling-dominated, so point and per-mode
    // preconditioners are GUARANTEED to diverge; only the full NP x NP block
    // converges). MatSetBlockSize makes GAMG aggregate mode-blocks and smooth
    // block-wise -- i.e. point-block-Jacobi as the MG smoother.
    PetscErrorCode AssemblePmat() {
        PetscFunctionBeginUser;
        const PetscInt nc = cE - cS;
        if (!Pmat) {
            PetscCall(MatCreate(PETSC_COMM_WORLD, &Pmat));
            PetscCall(MatSetSizes(Pmat, nc * NP, nc * NP, PETSC_DETERMINE, PETSC_DETERMINE));
            PetscCall(MatSetType(Pmat, MATAIJ));
            PetscCall(MatSetBlockSize(Pmat, NP));            // <- the LFA requirement
            PetscCall(MatSeqAIJSetPreallocation(Pmat, 24 * NP, NULL));
            PetscCall(MatMPIAIJSetPreallocation(Pmat, 24 * NP, NULL, 24 * NP, NULL));
            PetscCall(MatSetOption(Pmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        }
        PetscCall(MatZeroEntries(Pmat));

        const PetscScalar *x, *ca;
        PetscCall(VecGetArrayRead(X, &x));
        PetscCall(VecGetArrayRead(CA, &ca));
        for (PetscInt c = cS; c < cE; ++c) {
            const PetscScalar *qc, *ac;
            PetscCall(DMPlexPointGlobalRead(dmQ, c, x, &qc));
            PetscCall(DMPlexPointGlobalRead(dmCA, c, ca, &ac));
            if (!qc || !ac) continue;
            // the blocks read the FULL NF state; P entries are irrelevant (the
            // operator is affine in P, so its coefficients do not depend on P)
            Real Q[NF];
            for (int i = 0; i < NS; ++i) Q[i] = qc[i];
            for (int j = 0; j < NP; ++j) Q[VAM_P_IDX[j]] = 0.0;
            Real B[VAM_N_PBLOCK * NP * NP];
            vam_pressure_blocks<Real>(Q, ac, parameters.data(), (Real)dt_chorin, B);

            const PetscInt r = c - cS;
            for (int blk = 0; blk < VAM_N_PBLOCK; ++blk) {
                const Real *Bb = &B[blk * NP * NP];
                Mat S = Sblk[blk];
                if (!S) {                                    // A0: stencil is I
                    for (int k = 0; k < NP; ++k)
                        for (int l = 0; l < NP; ++l)
                            PetscCall(MatSetValue(Pmat, r * NP + k, r * NP + l,
                                                  Bb[k * NP + l], ADD_VALUES));
                    continue;
                }
                PetscInt ncols; const PetscInt *cols; const PetscScalar *vals;
                PetscCall(MatGetRow(S, r, &ncols, &cols, &vals));
                for (PetscInt j = 0; j < ncols; ++j) {
                    if (vals[j] == 0.0) continue;
                    for (int k = 0; k < NP; ++k)
                        for (int l = 0; l < NP; ++l) {
                            const PetscScalar v = Bb[k * NP + l] * vals[j];
                            if (v != 0.0)
                                PetscCall(MatSetValue(Pmat, r * NP + k, cols[j] * NP + l,
                                                      v, ADD_VALUES));
                        }
                }
                PetscCall(MatRestoreRow(S, r, &ncols, &cols, &vals));
            }
        }
        PetscCall(VecRestoreArrayRead(X, &x));
        PetscCall(VecRestoreArrayRead(CA, &ca));
        PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // CONSISTENCY GATE: the assembled Pmat and the matrix-free residual must be
    // the SAME operator, or the "preconditioner" is preconditioning a different
    // problem and the Krylov solve silently converges to the wrong answer.
    //   residual(P) - residual(0) == A*P  (the operator is affine)
    // so compare Pmat*P against that for a random P. Runs once, opt-in.
    PetscErrorCode VerifyPmat() {
        PetscFunctionBeginUser;
        Vec p, lhs, rhs, r0;
        PetscCall(VecDuplicate(Pv, &p));
        PetscCall(VecDuplicate(Pv, &lhs));
        PetscCall(VecDuplicate(Pv, &rhs));
        PetscCall(VecDuplicate(Pv, &r0));
        PetscRandom rnd;
        PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
        PetscCall(PetscRandomSetInterval(rnd, -1.0, 1.0));
        PetscCall(VecSetRandom(p, rnd));
        PetscCall(PetscRandomDestroy(&rnd));

        PetscCall(VecZeroEntries(lhs));
        PetscCall(PressureResidual(lhs, r0));       // r0 = residual(0)
        PetscCall(PressureResidual(p, rhs));        // rhs = residual(P)
        PetscCall(VecAXPY(rhs, -1.0, r0));          // rhs = A*P (matrix-free truth)
        PetscCall(MatMult(Pmat, p, lhs));           // lhs = Pmat*P (assembled)
        PetscReal nt, nd, nl;
        PetscCall(VecNorm(rhs, NORM_2, &nt));       // ||A*P|| matrix-free
        PetscCall(VecNorm(lhs, NORM_2, &nl));       // ||Pmat*P|| assembled
        PetscCall(VecAXPY(lhs, -1.0, rhs));
        PetscCall(VecNorm(lhs, NORM_2, &nd));
        if (rank == 0) {
            std::cout << "[Pmat] |Pmat*P| = " << nl << "   |A*P| = " << nt
                      << "   |diff| = " << nd
                      << "   rel = " << (nt > 0 ? nd / nt : nd) << std::endl;
            // A zero operator satisfies "Pmat*P == A*P" trivially. That is how
            // this check first passed with a PERFECT 0: it ran at setup, where
            // dt_chorin == 0, so both sides were identically zero. Refuse to
            // report a pass the test could not have failed.
            if (nt == 0.0 || nl == 0.0)
                std::cout << "[Pmat] ⚠ VACUOUS: an operand is identically zero "
                             "(dt=" << dt_chorin << ") — this check proves NOTHING."
                          << std::endl;
            else if (nd / nt < 1e-10)
                std::cout << "[Pmat] PASS: assembled operator == matrix-free operator" << std::endl;
            else
                std::cout << "[Pmat] 🔴 FAIL: the assembled Pmat is NOT the same "
                             "operator as the matrix-free residual" << std::endl;
        }
        PetscCall(VecDestroy(&p)); PetscCall(VecDestroy(&lhs));
        PetscCall(VecDestroy(&rhs)); PetscCall(VecDestroy(&r0));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupChorin() {
        PetscFunctionBeginUser;
        PetscCall(CloneCellDM(NP, &dmP));
        PetscCall(CloneCellDM(NCA, &dmCA));
        PetscCall(DMCreateGlobalVector(dmP, &Pv));
        PetscCall(VecDuplicate(Pv, &Rhs));
        PetscCall(VecDuplicate(Pv, &R0));
        PetscCall(VecDuplicate(Pv, &Pcur));
        PetscCall(DMCreateGlobalVector(dmCA, &CA));
        PetscInt n; PetscCall(VecGetLocalSize(Pv, &n));
        PetscCall(MatCreateShell(PetscObjectComm((PetscObject)dmQ), n, n, PETSC_DECIDE, PETSC_DECIDE, this, &Aop));
        PetscCall(MatShellSetOperation(Aop, MATOP_MULT, (void(*)(void))MatMult_Pressure));
        PetscCall(KSPCreate(PetscObjectComm((PetscObject)dmQ), &ksp));
        PetscCall(KSPSetOperators(ksp, Aop, Aop));
        PetscCall(KSPSetType(ksp, KSPGMRES));
        PetscCall(KSPGMRESSetRestart(ksp, 40));

        // ── REQ-172: block-aware PCGAMG on an ASSEMBLED Pmat ──────────────
        // BACKGROUND (measured, deliverable-4 sweep): with PCNONE the GMRES
        // iterations grow with the mesh -- 240:144, 960:237, 3840:400 -- and at
        // 3840 every solve returns DIVERGED_ITS at the cap, so the pressure is
        // NOT converged and results there are INVALID, not merely slow. It is
        // also why per-cell cost RISES with N (33.5 -> 216.7 ms/step for 4x
        // cells = N^1.35). @amrex hit the SAME wall at 1320 cells with entirely
        // different linear algebra ⇒ the property is in the VAM pressure
        // OPERATOR, not in either backend's solver.
        //   @amrex's LFA says WHICH fix can work: rho(D^-1 M) = 3.9 (modal) /
        // 1.8 (nodal), both > 1 ⇒ the NP x NP mode block is COUPLING-dominated,
        // so point-Jacobi and per-mode multigrid are GUARANTEED to diverge (they
        // measured both); only point-BLOCK-Jacobi converges, but it is O(N).
        //   ⇒ the Pmat is assembled WITH BLOCK SIZE NP so GAMG aggregates
        // mode-blocks and smooths block-wise -- point-block-Jacobi as the MG
        // SMOOTHER, the one combination that both converges and can be
        // mesh-independent. A scalar/per-field PC here would BE their diverging
        // per-mode MG. Aop (MatShell) stays the Amat = the true matvec; Pmat is
        // only the preconditioner operand.
        if (getenv("ZOOMY_VAM_PCNONE")) use_gamg = false;   // A/B against the old path
        PC pc; PetscCall(KSPGetPC(ksp, &pc));
        if (use_gamg) {
            for (int blk = 0; blk < VAM_N_PBLOCK; ++blk) {
                const VamPBlockSpec &s = VAM_PBLOCKS[blk];
                if (s.ax1 < 0) { Sblk[blk] = nullptr; continue; }   // A0 -> identity
                Mat D1; PetscCall(BuildDeriv1Mat(s.ax1, &D1));
                if (s.ax2 < 0) { Sblk[blk] = D1; continue; }
                Mat D2, prod;
                PetscCall(BuildDeriv1Mat(s.ax2, &D2));
                // second derivative = the SAME Deriv1 chain FillCA applies
                PetscCall(MatMatMult(D2, D1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &prod));
                PetscCall(MatDestroy(&D1)); PetscCall(MatDestroy(&D2));
                Sblk[blk] = prod;
            }
            PetscCall(AssemblePmat());
            PetscCall(KSPSetOperators(ksp, Aop, Pmat));
            PetscCall(PCSetType(pc, PCGAMG));
        } else {
            PetscCall(PCSetType(pc, PCNONE));
        }
        PetscCall(KSPSetTolerances(ksp, 1e-9, 1e-12, PETSC_DEFAULT, 400));
        PetscCall(KSPSetFromOptions(ksp));
        PetscCall(VecZeroEntries(Pv));
        // NOTE: VerifyPmat is deliberately NOT called here. At SetupChorin time
        // dt_chorin is still 0 and CA is unfilled, and EVERY pressure block
        // carries a dt factor -> Pmat == 0 and residual(P)-residual(0) == 0, so
        // the check compares zero to zero and reports a perfect 0 error. It is
        // run from the step loop instead, after the first Predictor +
        // ComputeFrozenAux, where dt and the frozen state are real.
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode TeardownChorin() {
        PetscFunctionBeginUser;
        if (ksp) PetscCall(KSPDestroy(&ksp));
        if (Aop) PetscCall(MatDestroy(&Aop));
        if (Pmat) PetscCall(MatDestroy(&Pmat));
        for (int i = 0; i < VAM_N_PBLOCK; ++i) if (Sblk[i]) PetscCall(MatDestroy(&Sblk[i]));
        for (Vec *v : {&Pv,&Rhs,&R0,&CA,&Pcur}) if (*v) PetscCall(VecDestroy(v));
        if (dmP) PetscCall(DMDestroy(&dmP));
        if (dmCA) PetscCall(DMDestroy(&dmCA));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // predictor: forward-Euler on the pressure-free FV RHS + bed-trace source.
    PetscErrorCode Predictor(PetscReal dt) {
        PetscFunctionBeginUser;
        Vec F; PetscCall(DMGetGlobalVector(dmQ, &F));
        PetscCall(transport->FormRHS(0.0, X, F));
        PetscCall(VecAXPY(X, dt, F));
        PetscCall(DMRestoreGlobalVector(dmQ, &F));
        if (source_solver) PetscCall(source_solver->Solve(dt, X, A));
        PetscCall(EnforcePhysicalConstraints(X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // De-interleave field `idx` (dof `ndof`) of Vec Vg over DM dm into a scalar array.
    PetscErrorCode ExtractField(DM dm, Vec Vg, PetscInt idx, std::vector<Real>& out) {
        PetscFunctionBeginUser;
        out.assign(cE - cS, 0.0);
        const PetscScalar *x; PetscCall(VecGetArrayRead(Vg, &x));
        for (PetscInt c = cS; c < cE; ++c) {
            const PetscScalar *q; PetscCall(DMPlexPointGlobalRead(dm, c, x, &q));
            if (q) out[c - cS] = (Real)q[idx];
        }
        PetscCall(VecRestoreArrayRead(Vg, &x));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscErrorCode StoreCA(PetscInt caidx, const std::vector<Real>& fld) {
        PetscFunctionBeginUser;
        PetscScalar *ca; PetscCall(VecGetArray(CA, &ca));
        for (PetscInt c = cS; c < cE; ++c) {
            PetscScalar *a; PetscCall(DMPlexPointGlobalRef(dmCA, c, ca, &a));
            if (a) a[caidx] = fld[c - cS];
        }
        PetscCall(VecRestoreArray(CA, &ca));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    void Deriv1(const std::vector<Real>& fld, int axis, std::vector<Real>& out) {
        ZoomyMesh mesh{dmQ, cS, cE};
        out.assign(cE - cS, 0.0);
        compute_derivative<Real>(out.data(), fld.data(), axis==0?1:0, axis==1?1:0, 0, mesh);
    }

    // Fill the CA slots selected by `want_P`, driven ENTIRELY by the generated
    // VAM_CA table (never by a hand-written index list: the layout is
    // name->index and changes with `dimension` — a dim=3 table writes P0_y into
    // dim=2's P_1_x slot, silently).
    //   want_P=false -> frozen state aux (b/h/q), read from X on dmQ, once/step
    //   want_P=true  -> pressure aux, read from the given P vector on dmP,
    //                   recomputed every KSP matvec from the current iterate
    PetscErrorCode FillCA(bool want_P, Vec Pvec) {
        PetscFunctionBeginUser;
        std::vector<Real> f, d, dd;
        for (int j = 0; j < NCA; ++j) {
            const VamAuxSpec &s = VAM_CA[j];
            if (s.is_P != want_P) continue;
            if (want_P) {
                // s.field indexes the FULL state; map it to the P vector's own
                // component (P_0 -> 0, P_1 -> 1) via VAM_P_IDX.
                int pc = -1;
                for (int k = 0; k < NP; ++k) if (VAM_P_IDX[k] == s.field) pc = k;
                if (pc < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB,
                                    "CA[%d] flagged is_P but field %d is not a pressure mode", j, s.field);
                PetscCall(ExtractField(dmP, Pvec, pc, f));
            } else {
                // frozen fields are always evolved-state slots (field < NS)
                if (s.field >= NS) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB,
                                           "CA[%d] frozen field %d >= NS=%d", j, s.field, NS);
                PetscCall(ExtractField(dmQ, X, s.field, f));
            }
            Deriv1(f, s.ax1, d);
            if (s.ax2 >= 0) { Deriv1(d, s.ax2, dd); PetscCall(StoreCA(j, dd)); }  // mixed ax1!=ax2 works
            else            { PetscCall(StoreCA(j, d)); }
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscErrorCode ComputeFrozenAux()   { return FillCA(false, nullptr); }
    PetscErrorCode ComputePAux(Vec P)   { return FillCA(true,  P); }

    // residual(P) into `out` (NP/cell): refresh P-aux, assemble Q10 per cell, eval.
    PetscErrorCode PressureResidual(Vec P, Vec out) {
        PetscFunctionBeginUser;
        PetscCall(ComputePAux(P));
        const PetscScalar *x, *pp, *ca; PetscScalar *o;
        PetscCall(VecGetArrayRead(X, &x)); PetscCall(VecGetArrayRead(P, &pp));
        PetscCall(VecGetArrayRead(CA, &ca)); PetscCall(VecGetArray(out, &o));
        for (PetscInt c = cS; c < cE; ++c) {
            const PetscScalar *q8, *pc, *a; PetscScalar *r;
            PetscCall(DMPlexPointGlobalRead(dmQ, c, x, &q8));
            PetscCall(DMPlexPointGlobalRead(dmP, c, pp, &pc));
            PetscCall(DMPlexPointGlobalRead(dmCA, c, ca, &a));
            PetscCall(DMPlexPointGlobalRef(dmP, c, o, &r));
            if (q8 && pc && a && r) {
                // assemble the full NF-state the generated ops expect: X then P
                Real Q[NF];
                for (int i = 0; i < NS; ++i) Q[i] = q8[i];
                for (int j = 0; j < NP; ++j) Q[VAM_P_IDX[j]] = pc[j];
                Real res[NP];
                vam_pressure_residual<Real>(Q, a, parameters.data(), (Real)dt_chorin, res);
                for (int j = 0; j < NP; ++j) r[j] = res[j];
            }
        }
        PetscCall(VecRestoreArrayRead(X, &x)); PetscCall(VecRestoreArrayRead(P, &pp));
        PetscCall(VecRestoreArrayRead(CA, &ca)); PetscCall(VecRestoreArray(out, &o));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    // MatShell: y = A*x = residual(x) - R0   (affine block).
    static PetscErrorCode MatMult_Pressure(Mat M, Vec x, Vec y) {
        PetscFunctionBeginUser;
        ChorinVAMSolver *s; PetscCall(MatShellGetContext(M, &s));
        PetscCall(s->PressureResidual(x, y));
        PetscCall(VecAXPY(y, -1.0, s->R0));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscErrorCode PressureSolve() {
        PetscFunctionBeginUser;
        // The blocks carry dt AND the frozen predictor state (h, b_x, h_x, ...),
        // both of which change every step, so the Pmat must be re-assembled --
        // a stale Pmat is still a valid preconditioner (it only slows GMRES) but
        // it drifts from the operator and costs iterations.
        if (use_gamg) PetscCall(AssemblePmat());
        PetscCall(VecZeroEntries(Pcur));
        PetscCall(PressureResidual(Pcur, R0));       // R0 = residual(P=0)
        PetscCall(VecCopy(R0, Rhs)); PetscCall(VecScale(Rhs, -1.0));
        PetscCall(KSPSolve(ksp, Rhs, Pv));           // A P = -R0
        KSPConvergedReason reason; PetscCall(KSPGetConvergedReason(ksp, &reason));
        PetscInt its; PetscCall(KSPGetIterationNumber(ksp, &its));
        if (rank == 0 && reason < 0)
            std::cout << "[KSP] pressure NOT converged (reason " << reason << ", " << its << " its)" << std::endl;
        // TEMP DIAGNOSTIC (REQ-172): is the pressure solve the CAUSE or the
        // VICTIM? Print ||P||, ||rhs||, the Pmat inf-norm and the state's h_min
        // each step. If h_min degrades BEFORE ||P|| blows up the pressure is a
        // victim; if ||P|| explodes first the operator is the problem.
        if (getenv("ZOOMY_VAM_DIAG")) {
            PetscReal np_, nr_, hmin = 1e300, pn = -1;
            PetscCall(VecNorm(Pv, NORM_2, &np_));
            PetscCall(VecNorm(Rhs, NORM_2, &nr_));
            if (Pmat) PetscCall(MatNorm(Pmat, NORM_INFINITY, &pn));
            const PetscScalar *xa; PetscCall(VecGetArrayRead(X, &xa));
            for (PetscInt c = cS; c < cE; ++c) {
                const PetscScalar *qc; PetscCall(DMPlexPointGlobalRead(dmQ, c, xa, &qc));
                if (qc && qc[1] < hmin) hmin = qc[1];
            }
            PetscCall(VecRestoreArrayRead(X, &xa));
            if (rank == 0)
                std::cout << "[DIAG] reason=" << reason << " its=" << its
                          << "  |P|=" << np_ << "  |rhs|=" << nr_
                          << "  |Pmat|_inf=" << pn << "  h_min=" << hmin << std::endl;
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    // corrector: refresh P-grad from the solved Pv, apply vam_corrector to the
    // generated VAM_CORR_IDX slots (dim=2: q_0,q_1,r_0,r_1; dim=3: +q_y modes).
    PetscErrorCode Corrector() {
        PetscFunctionBeginUser;
        PetscCall(ComputePAux(Pv));
        const PetscScalar *pp, *ca; PetscScalar *x;
        PetscCall(VecGetArrayRead(Pv, &pp)); PetscCall(VecGetArrayRead(CA, &ca)); PetscCall(VecGetArray(X, &x));
        for (PetscInt c = cS; c < cE; ++c) {
            PetscScalar *q8; const PetscScalar *pc, *a;
            PetscCall(DMPlexPointGlobalRef(dmQ, c, x, &q8));
            PetscCall(DMPlexPointGlobalRead(dmP, c, pp, &pc));
            PetscCall(DMPlexPointGlobalRead(dmCA, c, ca, &a));
            if (q8 && pc && a) {
                Real Q[NF];
                for (int i = 0; i < NS; ++i) Q[i] = q8[i];
                for (int j = 0; j < NP; ++j) Q[VAM_P_IDX[j]] = pc[j];
                Real upd[VAM_N_CORR];
                vam_corrector<Real>(Q, a, parameters.data(), (Real)dt_chorin, upd);
                for (int j = 0; j < VAM_N_CORR; ++j) q8[VAM_CORR_IDX[j]] = upd[j];
            }
        }
        PetscCall(VecRestoreArrayRead(Pv, &pp)); PetscCall(VecRestoreArrayRead(CA, &ca)); PetscCall(VecRestoreArray(X, &x));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

#endif // ZOOMY_MODEL_IS_VAM

#endif // CHORIN_VAM_HPP
