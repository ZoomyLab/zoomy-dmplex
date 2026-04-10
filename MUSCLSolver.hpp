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
        PetscCall(InitializeComponents());

        std::vector<std::string> names = {"b", "h", "u", "v", "w", "p"};
        PetscCall(io->Setup3D(dmQ, 6, names));
        PetscCall(this->SetupInitialConditions());

        if (!strategy) strategy = std::make_shared<SplittingStrategy>();
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts));

        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

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

        // ── Main time loop (single-pass, no detection/rollback) ──
        while (time < settings.solver.t_end) {
            PetscCall(TSStep(ts));
            PetscCall(EnforcePhysicalConstraints(X));

            step_num++;
            PetscCall(TSGetTime(ts, &time));
            PetscCall(PostStep(ts));
            PetscCall(MonitorWrapper(ts, step_num, time, X, this));
        }

        if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
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
        PetscCall(DMRestoreLocalVector(dmQ, &U_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

#endif // MUSCL_SOLVER_HPP
