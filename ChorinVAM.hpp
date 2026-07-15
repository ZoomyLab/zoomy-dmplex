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
        PC pc; PetscCall(KSPGetPC(ksp, &pc)); PetscCall(PCSetType(pc, PCNONE));
        PetscCall(KSPSetTolerances(ksp, 1e-9, 1e-12, PETSC_DEFAULT, 400));
        PetscCall(KSPSetFromOptions(ksp));
        PetscCall(VecZeroEntries(Pv));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode TeardownChorin() {
        PetscFunctionBeginUser;
        if (ksp) PetscCall(KSPDestroy(&ksp));
        if (Aop) PetscCall(MatDestroy(&Aop));
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
        PetscCall(VecZeroEntries(Pcur));
        PetscCall(PressureResidual(Pcur, R0));       // R0 = residual(P=0)
        PetscCall(VecCopy(R0, Rhs)); PetscCall(VecScale(Rhs, -1.0));
        PetscCall(KSPSolve(ksp, Rhs, Pv));           // A P = -R0
        KSPConvergedReason reason; PetscCall(KSPGetConvergedReason(ksp, &reason));
        PetscInt its; PetscCall(KSPGetIterationNumber(ksp, &its));
        if (rank == 0 && reason < 0)
            std::cout << "[KSP] pressure NOT converged (reason " << reason << ", " << its << " its)" << std::endl;
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
