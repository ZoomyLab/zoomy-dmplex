#ifndef SOLVER_STRATEGIES_HPP
#define SOLVER_STRATEGIES_HPP

#include <petsc.h>
#include "ModularSolver.hpp"

// Abstract Interface
class SolverStrategy {
public:
    virtual ~SolverStrategy() = default;
    virtual PetscErrorCode SetupTS(TS ts, ModularSolver* solver) = 0;
};

// ============================================================================
// 1. Splitting Strategy
// ============================================================================
class SplittingStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetRHSFunction(ts, NULL, RHSWrapper, solver));
        PetscCall(TSSetPostStep(ts, SplittingWrapper));
        // First-order space -> forward Euler (RK1), matching the jax/numpy
        // reference: the Audusse-HR cell-mean positivity (Xing-Zhang) is a
        // forward-Euler statement; a multi-stage SSP integrator's intermediate
        // stages are not convex-combination-positive. Order >= 2 keeps SSP.
        if (solver->settings.solver.reconstruction_order >= 2) {
            // SSP-RK2 (2-stage Heun): 2nd-order in time matches the 2nd-order
            // space, SSP, and is what the jax reference uses. The old 10-stage
            // SSP-RK104 is 4th-order in time -- overkill here (10 flux evals vs 2);
            // its only benefit is a larger SSP coefficient (~6 vs 1) for high-CFL
            // positivity, but we run CFL 0.45 (well under Heun's limit) and MOOD
            // handles positivity a-posteriori, so the extra stages are wasted.
            PetscCall(TSSetType(ts, TSSSP));
            PetscCall(TSSSPSetType(ts, TSSSPRKS2));
            PetscCall(TSSSPSetNumStages(ts, solver->settings.solver.ssp_stages));
        } else {
            PetscCall(TSSetType(ts, TSEULER));
        }
        return PETSC_SUCCESS;
    }

private:
    static PetscErrorCode RHSWrapper(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
        return ((ModularSolver*)ctx)->transport->FormRHS(t, X, F);
    }

    static PetscErrorCode SplittingWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        ModularSolver* solver = (ModularSolver*)ctx;
        Vec X_curr; TSGetSolution(ts, &X_curr);
        PetscReal dt; TSGetTimeStep(ts, &dt); 
        if(solver->source_solver) {
             PetscCall(solver->source_solver->Solve(dt, X_curr, solver->A)); 
        }
        return solver->PostStep(ts);
    }
};

// ============================================================================
// 2. IMEX Strategy
// ============================================================================
class IMEXStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetRHSFunction(ts, NULL, RHSWrapper, solver));
        PetscCall(TSSetIFunction(ts, NULL, IFunctionWrapper, solver));
        
        Mat P;
        PetscCall(DMCreateMatrix(solver->dmQ, &P));
        // REQ-165: Amat must be P, not NULL. With Amat=NULL the KSP has no
        // assembled operator to MatMult and TSARKIMEX dies on the FIRST implicit
        // stage with "Object is in wrong state: Not for unassembled matrix"
        // (MatMult -> PCApplyBAorAB -> KSPGMRESCycle). ImplicitStrategy below
        // already passes (P, P); this path was simply never run. FormSourceJacobian
        // assembles P (ModularSolver.hpp:391/451), so the same matrix serves as
        // both operator and preconditioner.
        PetscCall(TSSetIJacobian(ts, P, P, IJacobianWrapper, solver));
        PetscCall(MatDestroy(&P));

        PetscCall(TSSetType(ts, TSARKIMEX)); 
        PetscCall(TSARKIMEXSetType(ts, "2e"));
        PetscCall(TSSetPostStep(ts, PostStepWrapper)); 
        return PETSC_SUCCESS;
    }

private:
    static PetscErrorCode PostStepWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        return ((ModularSolver*)ctx)->PostStep(ts);
    }

    static PetscErrorCode RHSWrapper(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
        return ((ModularSolver*)ctx)->transport->FormRHS(t, X, F);
    }

    static PetscErrorCode IFunctionWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, Vec F, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;
        PetscCall(VecCopy(X_dot, F));
        PetscCall(solver->AddImplicitSourceToResidual(X, F, -1.0));
        return PETSC_SUCCESS;
    }

    static PetscErrorCode IJacobianWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, PetscReal a, Mat J, Mat P, void* ctx) {
        return ((ModularSolver*)ctx)->FormSourceJacobian(t, X, a, P);
    }
};

// ============================================================================
// 3. Fully Implicit Strategy
// ============================================================================
class FullyImplicitStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetType(ts, TSBDF));
        PetscCall(TSBDFSetOrder(ts, 2));

        PetscCall(TSSetIFunction(ts, NULL, IFunctionWrapper, solver));
        
        Mat P;
        PetscCall(DMCreateMatrix(solver->dmQ, &P));
        
        // --- Safety: Allow new nonzeros if the preallocation is slightly off ---
        PetscCall(MatSetOption(P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE)); 

        PetscCall(TSSetIJacobian(ts, P, P, IJacobianWrapper, solver));
        PetscCall(MatDestroy(&P));

        SNES snes;
        PetscCall(TSGetSNES(ts, &snes));
        PetscCall(SNESSetUseMatrixFree(snes, PETSC_TRUE, PETSC_FALSE));

        PetscCall(TSSetPostStep(ts, PostStepWrapper));
        return PETSC_SUCCESS;
    }

private:
    static PetscErrorCode PostStepWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        return ((ModularSolver*)ctx)->PostStep(ts);
    }

    static PetscErrorCode IFunctionWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, Vec F, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;
        PetscCall(VecCopy(X_dot, F));
        PetscCall(solver->AddImplicitSourceToResidual(X, F, -1.0));
        
        Vec R_transport;
        PetscCall(DMGetGlobalVector(solver->dmQ, &R_transport));
        PetscCall(solver->transport->FormRHS(t, X, R_transport));
        PetscCall(VecAXPY(F, -1.0, R_transport));
        PetscCall(DMRestoreGlobalVector(solver->dmQ, &R_transport));
        return PETSC_SUCCESS;
    }

    static PetscErrorCode IJacobianWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, PetscReal a, Mat J, Mat P, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;
        
        // Assemble Preconditioner P
        PetscCall(solver->FormImplicitJacobian(t, X, a, P));

        // Initialize Matrix-Free J
        if (J && J != P) {
            PetscBool is_mffd;
            PetscCall(PetscObjectTypeCompare((PetscObject)J, MATMFFD, &is_mffd));
            if (is_mffd) {
                PetscCall(MatMFFDSetBase(J, X, NULL));
            }
            PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
        }
        return PETSC_SUCCESS;
    }
};

#endif