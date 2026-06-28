#!/usr/bin/env python
"""Regenerate the zoomy_dmplex C++ headers (``Model.H`` / ``Numerics.H``) from a
**NumericalSystemModel**, the proper pipeline (mirrors malpasset_sme_jax.py).

Key points (why an NSM, not the raw model):
  * The raw ``SME`` flux carries a bare ``1/h`` and blows up on a dry bed. The
    correct model is desingularized: ``MalpassetSME(desingularize=True)`` carries
    a KP-regularized ``hinv`` aux (``hinv = √2·h/√(h⁴+max(h,eps)⁴)`` ≈ 1/h for
    h≥eps, finite at h→0) and the conservative flux uses ``hinv`` instead of
    ``1/h``. This is the "use a NumericalSystemModel that regularizes 1/h" path.
  * ``NumericalSystemModel.from_system_model(sm, reconstruction=…, riemann=
    PositiveNonconservativeRusanov)`` adds the well-balanced, positivity-
    preserving Riemann + reconstruction. The C++ printers normalize their input
    via ``to_numerical_system_model``, so the regularized NSM is what is lowered.

Note: ``MalpassetSME`` currently lives in the thesis Malpasset case
(``thesis/cases/malpasset_jax``); the KP ``hinv`` desingularization should ideally
be promoted to a reusable zoomy_core option (see ORGANIZATION.md request).

``compute_derivative`` (the spatial-derivative aux) is NOT consumed by SME(0)'s
flux/source/NCP/eigenvalues (those use only the local ``hinv``); a proper
mesh-aware field-level definition is requested from core (the per-cell C++ aux
emission cannot express a non-local derivative).

Usage:
    python create_model.py                  # both headers, MalpassetSME L0, order 1
    python create_model.py --level 1 --order 2
    python create_model.py --model sme      # plain (non-desingularized) SME
"""
import argparse
import sys
from pathlib import Path

from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation, FromModel)
from zoomy_core.transformation.to_c import CppModel, CppNumerics

HERE = Path(__file__).resolve().parent
# MalpassetSME (desingularized hinv) lives in the thesis Malpasset case.
_MALPASSET_CASE = (HERE.parent.parent / "thesis" / "cases" / "malpasset_jax")


def build_system_model(level, outer="wall", dimension=3, model="malpasset"):
    """Build the SystemModel with its boundary conditions.

    ``model='malpasset'`` uses the desingularized ``MalpassetSME`` (hinv flux);
    ``model='sme'`` uses the plain core ``SME`` (bare 1/h — dry-bed unstable).
    Tag name ``default`` is special in VirtualSolver: meshes without
    ``$PhysicalNames`` (the Malpasset gmsh) route ALL boundary faces to it.
    """
    bc = (FromModel(tag="default", definition="wall")
          if outer == "wall" else Extrapolation(tag="default"))
    bcs = BoundaryConditions([bc])
    if model == "malpasset":
        if str(_MALPASSET_CASE) not in sys.path:
            sys.path.insert(0, str(_MALPASSET_CASE))
        from sme_malpasset_model import MalpassetSME
        m = MalpassetSME(level=level, dimension=dimension,
                         boundary_conditions=bcs)
    else:
        from zoomy_core.model.models import SME
        m = SME(level=level, dimension=dimension, boundary_conditions=bcs)
    return m.system_model


def emit(level, out=HERE, outer="wall", what="both", dimension=3,
         order=1, model="malpasset"):
    sm = build_system_model(level, outer=outer, dimension=dimension, model=model)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order),
        riemann=PositiveNonconservativeRusanov)

    if what in ("both", "model"):
        code = CppModel(nsm).create_code()
        # INTERIM (core printer bug, see ORGANIZATION.md request): Min/Max with an
        # integer-literal arg emits e.g. `std::max(0, Q[1])` (int vs double) which
        # fails to compile. Cast the bare-0 literal to the real type until the
        # printer types numeric Min/Max literals itself.
        code = (code.replace("std::max(0, ", "std::max((T)0.0, ")
                    .replace("std::min(0, ", "std::min((T)0.0, "))
        (out / "Model.H").write_text(code)
        print(f"wrote {out / 'Model.H'}")
    if what in ("both", "numerics"):
        num = PositiveNonconservativeRusanov(model=nsm)
        (out / "Numerics.H").write_text(CppNumerics(num).create_code())
        print(f"wrote {out / 'Numerics.H'}")

    print(f"{model} SME(level={level}, dim={dimension}, order={order}) "
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
    ap.add_argument("--model", choices=["malpasset", "sme"], default="malpasset",
                    help="malpasset = desingularized hinv flux (stable on dry bed)")
    a = ap.parse_args()
    emit(a.level, a.out, outer=a.outer, what=a.what, dimension=a.dimension,
         order=a.order, model=a.model)
