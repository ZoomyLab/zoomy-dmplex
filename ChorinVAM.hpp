#ifndef CHORIN_VAM_HPP
#define CHORIN_VAM_HPP

#include "MUSCLSolver.hpp"
#include "ChorinOps.H"
#include <petscksp.h>

// ─────────────────────────────────────────────────────────────────────────
// VAM Chorin pressure-split solver (hyperbolic-elliptic).
//
//   predictor (pressure-free FV) -> pressure (global elliptic KSP, matrix-free
//   on the AFFINE operator) -> corrector (pointwise, uses grad P).
//
// The predictor NSM DROPS the pressure modes (no evolution eq), so the evolved
// state X is NS=8 dof: [b,h,q_x0,q_x1,q_y0,q_y1,r0,r1]. The pressure P (NP=2)
// lives in a SEPARATE vector Pv. The generated ChorinOps read the FULL 10-state
// Q = [X(8), P(2)] — assembled per cell here. The corrector updates X[2..7].
// Derivative-aux array CA (20/cell): CA[0..7]=P first+second x/y derivs
// (recomputed each matvec), CA[8..19]=frozen b/h(1st+2nd)+q(1st) (once/step).
// ─────────────────────────────────────────────────────────────────────────
class ChorinVAMSolver : public MUSCLSolver {
    static constexpr int NS = 8;      // evolved state dof (Model<Real>::n_dof_q)
    static constexpr int NP = 2;      // pressure modes
    static constexpr int NF = 10;     // full VAM state for ChorinOps
    static constexpr int NCA = 20;    // derivative-aux entries per cell
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

    // 12 frozen aux CA[8..19] from X (b=0,h=1,q_x0=2,q_x1=3,q_y0=4,q_y1=5).
    PetscErrorCode ComputeFrozenAux() {
        PetscFunctionBeginUser;
        std::vector<Real> f, d, dd;
        auto d1 = [&](int sidx, int ax, int ca){ (void)ExtractField(dmQ,X,sidx,f); Deriv1(f,ax,d); (void)StoreCA(ca,d); };
        auto d2 = [&](int sidx, int ax, int ca){ (void)ExtractField(dmQ,X,sidx,f); Deriv1(f,ax,d); Deriv1(d,ax,dd); (void)StoreCA(ca,dd); };
        d1(0,0,8); d2(0,0,9); d1(0,1,10); d2(0,1,11);      // b_x,b_xx,b_y,b_yy
        d1(1,0,12); d2(1,0,13); d1(1,1,14); d2(1,1,15);    // h_x,h_xx,h_y,h_yy
        d1(2,0,16); d1(3,0,17); d1(4,1,18); d1(5,1,19);    // q_x0_x,q_x1_x,q_y0_y,q_y1_y
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    // 8 P-derivative aux CA[0..7] from a pressure vector P (fields 0=P_0,1=P_1).
    PetscErrorCode ComputePAux(Vec P) {
        PetscFunctionBeginUser;
        std::vector<Real> f, d, dd;
        auto d1 = [&](int pidx, int ax, int ca){ (void)ExtractField(dmP,P,pidx,f); Deriv1(f,ax,d); (void)StoreCA(ca,d); };
        auto d2 = [&](int pidx, int ax, int ca){ (void)ExtractField(dmP,P,pidx,f); Deriv1(f,ax,d); Deriv1(d,ax,dd); (void)StoreCA(ca,dd); };
        d1(0,0,0); d2(0,0,1); d1(0,1,2); d2(0,1,3);        // P0_x,P0_xx,P0_y,P0_yy
        d1(1,0,4); d2(1,0,5); d1(1,1,6); d2(1,1,7);        // P1_x,P1_xx,P1_y,P1_yy
        PetscFunctionReturn(PETSC_SUCCESS);
    }

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
                Real Q[NF]; for (int i=0;i<NS;++i) Q[i]=q8[i]; Q[8]=pc[0]; Q[9]=pc[1];
                Real res[NP];
                vam_pressure_residual<Real>(Q, a, parameters.data(), (Real)dt_chorin, res);
                r[0] = res[0]; r[1] = res[1];
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
    // corrector: refresh P-grad from the solved Pv, apply vam_corrector to X[2..7].
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
                Real Q[NF]; for (int i=0;i<NS;++i) Q[i]=q8[i]; Q[8]=pc[0]; Q[9]=pc[1];
                Real upd[6];
                vam_corrector<Real>(Q, a, parameters.data(), (Real)dt_chorin, upd);
                for (int i = 0; i < 6; ++i) q8[2 + i] = upd[i];
            }
        }
        PetscCall(VecRestoreArrayRead(Pv, &pp)); PetscCall(VecRestoreArrayRead(CA, &ca)); PetscCall(VecRestoreArray(X, &x));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

#endif // CHORIN_VAM_HPP
