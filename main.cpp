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
        auto strategy = std::make_shared<FullyImplicitStrategy>();

        MUSCLSolver solver;
        solver.SetStrategy(strategy);
        solver.SetFluxKernel(Numerics<Real>::numerical_flux);
        solver.SetNonConsFluxKernel(Numerics<Real>::numerical_fluctuations);

        PetscCall(solver.Run(argc, argv));
    }

    return PetscFinalize();
}
