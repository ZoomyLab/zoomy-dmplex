#!/usr/bin/env python
"""Regenerate the zoomy_dmplex C++ headers (``Model.H`` / ``Numerics.H``) from a
declarative Zoomy model, the same way zoomy_foam's ``create_model.py`` does.

This is the *reproducible* generation path for the PETSc/DMPlex backend: it
consumes a SystemModel + its ``BoundaryConditions`` and lowers them to C++ via
the shared core printers (``zoomy_core.transformation.to_c``), so the backend
reuses all of the symbolic-core numerics (flux / NCP / source / eigenvalues /
reconstruction / Riemann) and the placeholder-function map (``conditional``,
``clamp_*``, ``max_wavespeed`` …) defined once in
``zoomy_core.transformation.generic_c``.

Until core REQ-19 lands (see ORGANIZATION.md), the C++ *Model* printer
(``CppModel``) is stale against the current SystemModel API — it accesses
``gradient_variables`` / ``_boundary_conditions`` which current models no longer
expose — so ``--what model`` will raise. The *Numerics* printer already works on
the current core, so ``--what numerics`` is usable today and is how we keep the
backend's numerics in sync with the core.

Usage:
    python create_model.py                  # both headers, SME level 0, wall outer
    python create_model.py --level 1        # add one moment
    python create_model.py --what numerics  # only Numerics.H (works pre-REQ-19)
"""
import argparse
from pathlib import Path

from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions,
    Extrapolation,
    FromModel,
)
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_c import CppModel, CppNumerics

HERE = Path(__file__).resolve().parent


def build_model(level, outer="wall", dimension=3):
    """Build the declarative model with its boundary conditions.

    ``dimension`` is the SME convention (n horizontal + 1 vertical): use 3 for a
    2-D run (state ``[b, h, q_x_0, q_y_0, …]``, spatial ``dimension=2`` in the
    header) and 2 for a 1-D run. The flux is rotationally invariant (normal-
    projected), so the same kernels drive an unstructured 2-D mesh via the
    per-face normal in the Riemann solver.

    ``outer`` selects the single physical boundary tag the bundled mesh exposes:
    a reflective ``wall`` (default) or zero-gradient ``extrapolation``. Add more
    tags here exactly like zoomy_foam does — the BCs lower to the indexed
    ``boundary_conditions(bc_idx, …)`` kernel automatically.
    """
    # Tag name "default" is special in VirtualSolver: when the mesh has no
    # $PhysicalNames (e.g. the Malpasset gmsh), the solver marks ALL boundary
    # faces and routes them to this single BC. Use it for closed-domain runs.
    outer_bc = (
        FromModel(tag="default", definition="wall")
        if outer == "wall"
        else Extrapolation(tag="default")
    )
    return SME(level=level, dimension=dimension,
               boundary_conditions=BoundaryConditions([outer_bc]))


def emit(level, out=HERE, outer="wall", what="both", dimension=3):
    model = build_model(level, outer=outer, dimension=dimension)
    sm = model.system_model

    if what in ("both", "model"):
        code = CppModel(model).create_code()
        (out / "Model.H").write_text(code)
        print(f"wrote {out / 'Model.H'}")

    if what in ("both", "numerics"):
        num = PositiveNonconservativeRusanov(model=sm)
        code = CppNumerics(num).create_code()
        (out / "Numerics.H").write_text(code)
        print(f"wrote {out / 'Numerics.H'}")

    print(
        f"SME(level={level}, dimension={dimension}, outer={outer}) "
        f"state={[str(s) for s in sm.state]}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument(
        "--outer", choices=["wall", "extrapolation"], default="wall"
    )
    ap.add_argument(
        "--what", choices=["both", "model", "numerics"], default="both"
    )
    ap.add_argument("--dimension", type=int, default=3,
                    help="SME dimension convention: 3 => 2-D run, 2 => 1-D")
    a = ap.parse_args()
    emit(a.level, a.out, outer=a.outer, what=a.what, dimension=a.dimension)
