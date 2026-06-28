#!/usr/bin/env python
"""Regenerate the zoomy_dmplex C++ headers (``Model.H`` / ``Numerics.H``) from a
**NumericalSystemModel** — the proper pipeline (mirrors malpasset_sme_jax.py).

Key points (why an NSM, not the raw model):
  * The raw ``SME`` flux carries a bare ``1/h`` and blows up on a dry bed. The
    NSM regularizes it AUTOMATICALLY: ``from_system_model`` runs
    ``default_operations()`` for shallow-water transport (state with ``h``) =
    ``[desingularize_hinv(), gate_eigenvalues_dry()]``, so the flux uses the KP
    ``hinv`` (``√2·h/√(h⁴+max(h,eps)⁴)`` ≈ 1/h, finite at h→0) and dry cells are
    eigenvalue-gated. No per-case opt-in, no thesis import (REQ-67, core fe2ed58).
  * ``NumericalSystemModel.from_system_model(sm, reconstruction=…, riemann=
    PositiveNonconservativeRusanov, regularization=…)`` adds the well-balanced,
    positivity-preserving Riemann + reconstruction. The C++ printers normalize
    their input via ``to_numerical_system_model``, so the regularized NSM is
    what is lowered.

``compute_derivative`` (the spatial-derivative aux) is emitted FIELD-LEVEL/
mesh-aware (REQ-65) and implemented in ``UserFunctions.H`` as a real DMPlex
Green-Gauss gradient; it is not consumed by SME(0)'s flux/source/eigenvalues
(those use only the local ``hinv``), so it is inert for level 0.

Usage:
    python create_model.py                       # both headers, SME L0, order 1
    python create_model.py --level 1 --order 2
    python create_model.py --no-desingularize    # plain SME (dry-bed unstable)
"""
import argparse
from pathlib import Path

from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation, FromModel)
from zoomy_core.transformation.to_c import CppModel, CppNumerics

HERE = Path(__file__).resolve().parent


def build_system_model(level, outer="wall", dimension=3):
    """Build the core SME SystemModel with its boundary conditions.

    Tag name ``default`` is special in VirtualSolver: meshes without
    ``$PhysicalNames`` (the Malpasset gmsh) route ALL boundary faces to it.
    The 1/h desingularization is applied by the NSM (``desingularize=True``),
    not by the model — see ``emit`` (REQ-67: it is now a reusable core knob, so
    no thesis ``MalpassetSME`` import is needed).
    """
    bc = (FromModel(tag="default", definition="wall")
          if outer == "wall" else Extrapolation(tag="default"))
    return SME(level=level, dimension=dimension,
               boundary_conditions=BoundaryConditions([bc])).system_model


def emit(level, out=HERE, outer="wall", what="both", dimension=3, order=1):
    sm = build_system_model(level, outer=outer, dimension=dimension)
    # from_system_model auto-runs default_operations() for shallow-water
    # transport systems (state with `h`): the KP 1/h `desingularize_hinv` +
    # `gate_eigenvalues_dry` — so the flux uses the regularized `hinv` and the
    # dry bed is stable, with no per-case opt-in (REQ-67, core fe2ed58).
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order),
        riemann=PositiveNonconservativeRusanov)

    if what in ("both", "model"):
        # REQ-66 (core ae1a2aa) types Min/Max literals — no interim cast needed.
        (out / "Model.H").write_text(CppModel(nsm).create_code())
        print(f"wrote {out / 'Model.H'}")
    if what in ("both", "numerics"):
        num = PositiveNonconservativeRusanov(model=nsm)
        (out / "Numerics.H").write_text(CppNumerics(num).create_code())
        print(f"wrote {out / 'Numerics.H'}")

    print(f"SME(level={level}, dim={dimension}, order={order}) "
          f"state={[str(s) for s in sm.state]} "
          f"aux={[str(s) for s in sm.aux_state]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--outer", choices=["wall", "extrapolation"], default="wall")
    ap.add_argument("--what", choices=["both", "model", "numerics"], default="both")
    ap.add_argument("--dimension", type=int, default=3,
                    help="SME convention: 3 => 2-D run, 2 => 1-D")
    ap.add_argument("--order", type=int, default=1, help="reconstruction order")
    a = ap.parse_args()
    emit(a.level, a.out, outer=a.outer, what=a.what, dimension=a.dimension,
         order=a.order)
