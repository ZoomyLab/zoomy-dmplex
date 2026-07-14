#include <petsc.h>
#include <iostream>
#include <memory>

#include "MUSCLSolver.hpp"
#include "ChorinVAM.hpp"
#include "SolverStrategies.hpp"

static char help[] = "Zoomy FVM Solver (MUSCL + Venkatakrishnan; Chorin VAM)\n";

int main(int argc, char **argv) {
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    {
        // Peek at settings to pick the solver family: "muscl" (explicit FV,
        // splitting/imex/implicit inside) or "chorin" (VAM hyperbolic-elliptic
        // pressure split).
        char spath[PETSC_MAX_PATH_LEN] = "settings.json";
        PetscCall(PetscOptionsGetString(NULL, NULL, "-settings", spath, sizeof(spath), NULL));
        Settings st = Settings::from_json(spath);

        std::unique_ptr<MUSCLSolver> solver;
        if (st.solver.method == "chorin") solver = std::make_unique<ChorinVAMSolver>();
        else                              solver = std::make_unique<MUSCLSolver>();

        solver->SetFluxKernel(Numerics<Real>::numerical_flux);
        solver->SetNonConsFluxKernel(Numerics<Real>::numerical_fluctuations);
        PetscCall(solver->Run(argc, argv));
    }

    return PetscFinalize();
}
