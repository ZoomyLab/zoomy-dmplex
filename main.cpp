#include <petsc.h>
#include <iostream>
#include <memory>

#include "MUSCLSolver.hpp"
#include "SolverStrategies.hpp"

static char help[] = "Zoomy FVM Solver (MUSCL + Venkatakrishnan)\n";

int main(int argc, char **argv) {
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    {
        // Time integrator is chosen from settings.json (solver.time_integration:
        // "splitting" | "imex" | "implicit"), selected inside MUSCLSolver::Run.
        // To force one from code instead, call solver.SetStrategy(...) here.
        MUSCLSolver solver;
        solver.SetFluxKernel(Numerics<Real>::numerical_flux);
        solver.SetNonConsFluxKernel(Numerics<Real>::numerical_fluctuations);

        PetscCall(solver.Run(argc, argv));
    }

    return PetscFinalize();
}
