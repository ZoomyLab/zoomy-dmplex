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
import sys
from pathlib import Path

import sympy as sp
from zoomy_core.misc.misc import ZArray
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.operations import desingularize_hinv
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, FromModel)
from zoomy_core.transformation.to_c import CppModel, CppNumerics

HERE = Path(__file__).resolve().parent


def _guard_eig_powers():
    """Op: guard fractional powers of h (e.g. sqrt(h**5)) in the eigenvalues
    with Max(.,0) so a transient h<0 doesn't NaN — WITHOUT the dry conditional
    gate. This is the ``ev_gate=False`` positivity recipe the numpy/jax Malpasset
    reference uses: ungated wave speeds keep the Rusanov dissipation correctly
    sized (the core default ``gate_eigenvalues_dry`` zeros λ at h<eps, which
    undersizes dissipation between two near-dry cells and breaks the Xing-Zhang
    cell-mean-positivity decomposition)."""
    def _op(sm):
        ev = sm.eigenvalues
        if ev is None:
            return
        h = next(s for s in sm.state if str(s) == "h")
        def _frac_pow_h(x):
            return (isinstance(x, sp.Pow) and x.exp.is_number
                    and not x.exp.is_integer and x.base.has(h))
        def _guard(e):
            return sp.sympify(e).replace(
                _frac_pow_h, lambda x: sp.Pow(sp.Max(x.base, sp.S.Zero), x.exp))
        sm.eigenvalues = ZArray([_guard(e) for e in ev]).reshape(*ev.shape)
    _op.name = "guard_eig_powers"
    return _op


class _UngatedNSM(NumericalSystemModel):
    """NSM whose default_operations keep the KP hinv desingularization but drop
    the dry eigenvalue GATE (replaced by the powers-guard) — matching the jax
    reference's ev_gate=False for structural wet/dry positivity."""
    def default_operations(self):
        if not self._is_transport_system():
            return []
        if not any(str(s) == "h" for s in self.state):
            return []
        return [desingularize_hinv(), _guard_eig_powers()]
# MalpassetSME (carries the wet/dry momentum CLAMP via update_variables, the
# positivity safety net the numpy/jax solver applies every step) lives in the
# thesis Malpasset case. Core SME has no clamp and MalpassetSWE won't lower to
# C++ (conditional+NDimArray printer bug) — so for a stable dry-bed run dmplex
# imports this until core exposes the clamp reusably (see ORGANIZATION.md req).
_MALPASSET_CASE = (HERE.parent.parent / "thesis" / "cases" / "malpasset_jax")


def build_system_model(dimension=3):
    """Build MalpassetSME with clamp + desingularization — the model the working
    jax order-1 Malpasset run uses. ``clamp=True`` emits ``update_variables``
    (caps |u|, zeros momentum below wet_dry_eps); ``desingularize=True`` gives
    the KP ``hinv`` flux. Tag ``default`` routes ALL exterior faces to one Wall
    BC (the Malpasset gmsh has no ``$PhysicalNames``).
    """
    if str(_MALPASSET_CASE) not in sys.path:
        sys.path.insert(0, str(_MALPASSET_CASE))
    from sme_malpasset_model import MalpassetSME
    m = MalpassetSME(level=0, dimension=dimension, clamp=True, desingularize=True,
                     boundary_conditions=BoundaryConditions(
                         [FromModel(tag="default", definition="wall")]))
    return m.system_model


def emit(out=HERE, what="both", dimension=3, order=0):
    sm = build_system_model(dimension=dimension)
    nsm = _UngatedNSM.from_system_model(
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

    print(f"MalpassetSME(clamp, desingularize, dim={dimension}, order={order}) "
          f"state={[str(s) for s in sm.state]} "
          f"aux={[str(s) for s in sm.aux_state]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--what", choices=["both", "model", "numerics"], default="both")
    ap.add_argument("--dimension", type=int, default=3,
                    help="SME convention: 3 => 2-D run")
    ap.add_argument("--order", type=int, default=0,
                    help="reconstruction order (0 = first-order FV, matches the "
                         "working jax DG0 reference)")
    a = ap.parse_args()
    emit(a.out, what=a.what, dimension=a.dimension, order=a.order)
