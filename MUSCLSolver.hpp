#ifndef MUSCL_SOLVER_HPP
#define MUSCL_SOLVER_HPP

#include "ModularSolver.hpp"
#include "SolverStrategies.hpp"

/**
 * MUSCL Solver — explicit time stepping with MUSCL reconstruction.
 *
 * Refactored from MOODSolver: keeps the same explicit time loop structure
 * but removes the a-posteriori detection/rollback cycle. Reconstruction
 * order and limiter are set once via settings.
 */
class MUSCLSolver : public ModularSolver {
public:
    MUSCLSolver() : ModularSolver() {}
    ~MUSCLSolver() override = default;

    PetscErrorCode RegisterCallbacks(TS) override { return PETSC_SUCCESS; }
    PetscErrorCode UpdateState(Vec, Vec) override { return PETSC_SUCCESS; }

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(VirtualSolver::Initialize(argc, argv));

        // Set reconstruction order from settings (1=PCM, 2=MUSCL+limiter)
        if (settings.solver.reconstruction_order >= 2) {
            SetReconstruction(LINEAR);
        } else {
            SetReconstruction(PCM);
        }
        // Zhang-Shu a-priori positivity holds only for CFL <= 1/(2k+1) = 1/6
        // (2D, k=1); auto-clamp so order>=2 stays provably h>=0 (matches jax).
        if (settings.solver.reconstruction_order >= 2 &&
            settings.solver.positivity == "zhang_shu" && settings.solver.cfl > 1.0/6.0) {
            if (rank == 0) std::cout << "[INFO] clamping CFL " << settings.solver.cfl
                                     << " -> 1/6 for zhang_shu positivity" << std::endl;
            settings.solver.cfl = 1.0/6.0;
        }
        // Limiter on unless explicitly disabled ("none"). The limiter *type*
        // (venkatakrishnan/tvd) is selected inside LinearReconstructor; only the
        // on/off toggle is wired here.
        SetLimiters(settings.solver.limiter != "none");
        PetscCall(InitializeComponents());

        // Output field names derived generically from the model state count
        // (the generated Model.H sets n_dof_q); no hard-coded SWE layout.
        std::vector<std::string> names;
        for (int i = 0; i < Model<Real>::n_dof_q; ++i) names.push_back("q" + std::to_string(i));
        PetscCall(io->Setup3D(dmQ, Model<Real>::n_dof_q, names));
        PetscCall(this->SetupInitialConditions());
        // Aux state is not loaded from the IC file; derive it from the loaded Q
        // (update_aux_variables) so it is valid for the first ComputeTimeStep.
        PetscCall(EnforcePhysicalConstraints(X));

        // Select the time integrator from settings (jax-style), unless one was
        // already set explicitly (e.g. from main.cpp). "splitting" is the explicit
        // default; "imex" turns on TSARKIMEX (implicit source); "implicit" is BDF2.
        if (!strategy) {
            const std::string& ti = settings.solver.time_integration;
            if (ti == "imex") {
                strategy = std::make_shared<IMEXStrategy>();
            } else if (ti == "implicit") {
                strategy = std::make_shared<FullyImplicitStrategy>();
            } else {
                if (ti != "splitting" && rank == 0) {
                    std::cout << "[WARN] unknown time_integration '" << ti
                              << "', falling back to 'splitting'." << std::endl;
                }
                strategy = std::make_shared<SplittingStrategy>();
            }
        }
        if (rank == 0) {
            std::cout << "[INFO] time_integration=" << settings.solver.time_integration
                      << std::endl;
        }
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts));

        // Wire the chosen strategy into the TS (sets RHS/IFunction/IJacobian and
        // the TS type). Must run after the application context is set, since the
        // strategy callbacks fetch the solver via TSGetApplicationContext, and
        // before TSSetFromOptions so command-line options can still override.
        PetscCall(strategy->SetupTS(ts, this));

        // dt is driven manually by the CFL condition (ComputeTimeStep); disable
        // the TS adaptive controller so it doesn't fight the fixed step (it
        // otherwise trips "bad hmax in TSAdaptChoose" near t_end).
        {
            TSAdapt adapt;
            PetscCall(TSGetAdapt(ts, &adapt));
            PetscCall(TSAdaptSetType(adapt, TSADAPTNONE));
        }

        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        // STEPOVER (not MATCHSTEP): the manual CFL loop below stops once
        // time >= t_end, so let the last step overshoot slightly rather than
        // have the SSP adapter shrink dt to land on t_end (which trips
        // "bad hmax in TSAdaptChoose").
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

        PetscReal dt_start = std::max(ComputeTimeStep(), settings.solver.min_dt);
        PetscCall(TSSetTimeStep(ts, dt_start));
        PetscCall(TSSetFromOptions(ts));

        // Force reasonable SNES tolerances for explicit/IMEX stepping
        SNES snes;
        PetscCall(TSGetSNES(ts, &snes));
        PetscCall(SNESSetTolerances(snes, 1e-4, 1e-3, 1e-50, 20, 100));

        if (rank == 0) {
            std::cout << "[INFO] Starting MUSCL Solver (order="
                      << settings.solver.reconstruction_order
                      << ", limiter=" << settings.solver.limiter
                      << ")..." << std::endl;
        }

        PetscReal time;
        PetscCall(TSGetTime(ts, &time));
        PetscInt step_num = 0;
        PetscCall(MonitorWrapper(ts, 0, time, X, this));

        // LOCAL MOOD a-posteriori positivity (CPU-appropriate; jax does this
        // globally because it is vectorized). Take the stable SSP-RK104 order-2
        // candidate at full CFL; the healthy domain keeps 2nd-order time. Any
        // cell with h<0 is OVERRIDDEN by a local first-order forward-Euler update
        // computed from the saved old state (PCM, only troubled-adjacent faces) —
        // 1st order ONLY for the troubled wet/dry cells, the rest stays 2nd order.
        const bool mood = (settings.solver.positivity == "mood");
        Vec X_save = NULL, F_mood = NULL; std::vector<uint8_t> mood_mask;
        if (mood) {
            PetscCall(VecDuplicate(X, &X_save));
            PetscCall(VecDuplicate(X, &F_mood));
            PetscInt cS, cE; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cS, &cE));
            mood_mask.assign(cE - cS, 0);
        }

        // ── Main time loop ──
        while (time < settings.solver.t_end) {
            PetscReal t0 = time, dt0; PetscCall(TSGetTimeStep(ts, &dt0));
            // Land EXACTLY on t_end: TS_EXACTFINALTIME_STEPOVER lets the last step
            // overshoot (t=0.109 for t_end=0.1), which skews cross-backend speed
            // comparisons (REQ-159). Clamp the final dt instead of using MATCHSTEP
            // (the SSP adapter trips "bad hmax in TSAdaptChoose" under MATCHSTEP).
            if (t0 + dt0 > settings.solver.t_end) {
                dt0 = settings.solver.t_end - t0;
                PetscCall(TSSetTimeStep(ts, dt0));
            }
            if (mood) { PetscCall(VecCopy(X, X_save)); std::fill(mood_mask.begin(), mood_mask.end(), 0); }

            PetscCall(TSStep(ts));
            PetscCall(EnforcePhysicalConstraints(X));

            if (mood) {
                // SINGLE pass: the local override only modifies troubled cells
                // (making them O1-positive) and never touches healthy neighbours,
                // so it cannot create a new troubled cell -> one step is enough.
                PetscInt nt = 0; PetscCall(DetectTroubled(X, mood_mask, &nt));
                if (nt > 0) {
                    if (rank == 0) std::cout << "[MOOD] step " << step_num << " troubled=" << nt << std::endl;
                    PetscCall(transport->FormRHSTroubledO1(t0, X_save, F_mood, mood_mask));
                    PetscCall(ApplyTroubledO1(X, X_save, F_mood, dt0, mood_mask));
                    PetscCall(EnforcePhysicalConstraints(X));
                    // diagnostic: confirm the override actually cured h<0
                    std::vector<uint8_t> chk(mood_mask.size(), 0); PetscInt nt2 = 0;
                    PetscCall(DetectTroubled(X, chk, &nt2));
                    if (nt2 > 0 && rank == 0)
                        std::cout << "[MOOD] step " << step_num << " STILL troubled=" << nt2 << " after override" << std::endl;
                }
            }

            // Operator-split SOURCE step. The SplittingStrategy registers its
            // source solve via TSSetPostStep(SplittingWrapper), but PETSc only
            // invokes that from TSSolve — this manual TSStep loop never does, so
            // the source (friction, moment excitation, bed slope) was silently
            // dropped. Apply it here after transport (+MOOD positivity). IMEX /
            // FullyImplicit carry the source in their IFunction, so skip those.
            if (settings.solver.time_integration == "splitting" && source_solver) {
                PetscCall(source_solver->Solve(dt0, X, A));
                PetscCall(EnforcePhysicalConstraints(X));
            }

            step_num++;
            PetscCall(TSGetTime(ts, &time));
            PetscCall(PostStep(ts));
            PetscCall(MonitorWrapper(ts, step_num, time, X, this));
        }
        if (X_save) PetscCall(VecDestroy(&X_save));
        if (F_mood) PetscCall(VecDestroy(&F_mood));

        if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

protected:   // (was private) — ChorinVAMSolver reuses EnforcePhysicalConstraints etc.
    // MOOD detector: OR troubled cells (h < -tol) into mask, return global count.
    PetscErrorCode DetectTroubled(Vec X_global, std::vector<uint8_t>& mask, PetscInt* count) {
        PetscFunctionBeginUser;
        const PetscScalar *x; PetscCall(VecGetArrayRead(X_global, &x));
        PetscInt cS, cE; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cS, &cE));
        PetscInt nt = 0;
        for (PetscInt c = cS; c < cE; ++c) {
            const PetscScalar *qc; PetscCall(DMPlexPointGlobalRead(dmQ, c, x, &qc));
            if (!qc) continue;                       // ghost / not owned
            if (qc[1] < -1.0e-10) { mask[c - cS] = 1; nt++; }   // h = component 1
        }
        PetscCall(VecRestoreArrayRead(X_global, &x));
        PetscInt gnt; MPI_Allreduce(&nt, &gnt, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
        *count = gnt;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Overwrite troubled cells with their local first-order forward-Euler update
    // X[c] = X_save[c] + dt * F_O1[c]. Untroubled cells keep the order-2 candidate.
    PetscErrorCode ApplyTroubledO1(Vec X, Vec X_save, Vec F, PetscReal dt, const std::vector<uint8_t>& mask) {
        PetscFunctionBeginUser;
        const PetscScalar *xs, *fp; PetscScalar *xx;
        PetscCall(VecGetArrayRead(X_save, &xs)); PetscCall(VecGetArrayRead(F, &fp)); PetscCall(VecGetArray(X, &xx));
        PetscInt cS, cE; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cS, &cE));
        for (PetscInt c = cS; c < cE; ++c) {
            if (mask[c - cS] != 1) continue;
            const PetscScalar *qs, *qf; PetscScalar *qx;
            PetscCall(DMPlexPointGlobalRead(dmQ, c, xs, &qs));
            PetscCall(DMPlexPointGlobalRead(dmQ, c, fp, &qf));
            PetscCall(DMPlexPointGlobalRef(dmQ, c, xx, &qx));
            if (!qs || !qf || !qx) continue;
            for (int i = 0; i < Model<Real>::n_dof_q; ++i) qx[i] = qs[i] + dt * qf[i];
        }
        PetscCall(VecRestoreArrayRead(X_save, &xs)); PetscCall(VecRestoreArrayRead(F, &fp)); PetscCall(VecRestoreArray(X, &xx));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode EnforcePhysicalConstraints(Vec U_global) {
        PetscFunctionBeginUser;
        Vec U_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &U_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, U_global, INSERT_VALUES, U_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, U_global, INSERT_VALUES, U_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(transport->UpdateState(U_loc, A_loc));
        PetscCall(DMLocalToGlobalBegin(dmQ, U_loc, INSERT_VALUES, U_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, U_loc, INSERT_VALUES, U_global));
        // Scatter the recomputed aux (e.g. hinv) back to the GLOBAL A, otherwise
        // ComputeTimeStep reads hinv=0 -> max eigenvalue 0 -> the dt=1e-3 fallback
        // fires every step (the run crawls). With A updated, dt is the true CFL.
        PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMRestoreLocalVector(dmQ, &U_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

#endif // MUSCL_SOLVER_HPP
