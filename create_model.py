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


def build_model(level, outer="wall"):
    """Build the declarative model with its boundary conditions.

    ``outer`` selects the single physical boundary tag the bundled ``mesh.geo``
    exposes: a reflective ``wall`` (default) or zero-gradient ``extrapolation``.
    Add more tags here exactly like zoomy_foam does — the BCs lower to the
    indexed-Piecewise ``boundary_conditions(bc_idx, …)`` kernel automatically.
    """
    outer_bc = (
        FromModel(tag="outer", definition="wall")
        if outer == "wall"
        else Extrapolation(tag="outer")
    )
    return SME(level=level, boundary_conditions=BoundaryConditions([outer_bc]))


def emit(level, out=HERE, outer="wall", what="both"):
    model = build_model(level, outer=outer)
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
        f"SME(level={level}, outer={outer}) "
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
    a = ap.parse_args()
    emit(a.level, a.out, outer=a.outer, what=a.what)
