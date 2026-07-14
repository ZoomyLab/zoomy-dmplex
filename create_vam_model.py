#!/usr/bin/env python
"""Generate the VAM Chorin sub-model C++ headers for the dmplex backend.

VAM is hyperbolic-elliptic: VAM.chorin_split(dt) yields 3 SystemModels that
SHARE the full state [b,h,q_*,r_*,P_*]:
  * SM_pred  — hyperbolic predictor (pressure-free/hydrostatic): flux/NCP/
               eigenvalues + a bed-trace source. Lowered like any FV model.
  * SM_press — the elliptic pressure stage (affine A*P + R0); pressure_operator()
               gives analytic A0/Ax/Axx/RHS for a PETSc KSP assembly.
  * SM_corr  — pointwise corrector (update_variables).

WORKAROUND (until core adds it to the generic printer, cf. the amrex
`to_amrex._resolve_subs`): the VAM source carries `Subs(f(zeta),zeta,0)` bed-trace
nodes that the generic C printer emits verbatim (C has no `Subs`). We .doit() them
before lowering. Only Subs nodes are touched (opaque Galerkin brackets untouched).
"""
import sys, argparse
from pathlib import Path
import sympy as sp
from zoomy_core.misc.misc import ZArray
from create_model import _UngatedNSM
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import ReconstructionSpec
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_c import CppModel, CppNumerics

HERE = Path(__file__).resolve().parent


def resolve_subs(sm):
    # No-op since core REQ-130 (generic C printer resolves Subs(f(zeta),zeta,0)).
    # Kept as a defensive pass-through in case an old core is used.
    return sm


def build_split(level=1, dimension=3):
    m = VAM(level=level, dimension=dimension, closures=[Newtonian(), StressFree()])
    dt = sp.Symbol("dt", positive=True)
    return m.chorin_split(dt)


def emit_predictor(split, out=HERE, order=0):
    sm = resolve_subs(split.SM_pred)
    nsm = _UngatedNSM.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order),
        riemann=PositiveNonconservativeRusanov)
    (out / "Model.H").write_text(CppModel(nsm).create_code())
    (out / "Numerics.H").write_text(
        CppNumerics(PositiveNonconservativeRusanov(model=nsm)).create_code())
    print(f"predictor: {len(split.SM_pred.state)} states -> Model.H/Numerics.H")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--dimension", type=int, default=3)
    ap.add_argument("--order", type=int, default=0)
    a = ap.parse_args()
    split = build_split(a.level, a.dimension)
    emit_predictor(split, order=a.order)


def emit_chorin_ops(split, out=HERE):
    """Emit ChorinOps.H: the pressure residual (matrix-free A*P matvec) and the
    corrector, as standalone C++ free functions + the derivative-aux layout the
    solver must fill (1st via compute_derivative, 2nd via the Hessian stencil;
    P-derivatives recomputed each KSP matvec from the current P iterate)."""
    import sympy as sp
    from sympy.printing.c import ccode
    pr, ps, co = split.SM_pred, split.SM_press, split.SM_corr
    STATE = [str(s) for s in pr.state]; PARAM = [str(p) for p in pr.parameters]
    def scal(x):
        x = sp.sympify(x)
        while hasattr(x, "shape") and x.shape: x = x[0]
        return x
    press = [scal(x) for x in list(ps.source)]
    corr  = [scal(x) for x in list(co.update_variables)]
    def parse(nm):
        for fld in sorted(STATE, key=len, reverse=True):
            if nm.startswith(fld + "_"):
                rest = nm[len(fld)+1:].split("_")
                if all(t in ("x","y","z") for t in rest): return (fld, tuple(rest))
        return None
    syms = set()
    for e in press + corr: syms |= e.free_symbols
    derivs = sorted(nm for nm in (str(s) for s in syms)
                    if nm not in STATE and nm not in PARAM and nm != "dt" and parse(nm))
    didx = {nm: j for j, nm in enumerate(derivs)}
    # key on the ACTUAL symbol objects (assumptions differ from sp.Symbol(name);
    # xreplace must match the exact objects, else the raw names leak into the C).
    smap = {}
    for s in syms:
        nm = str(s)
        if nm in STATE:   smap[s] = sp.Symbol(f"Q[{STATE.index(nm)}]")
        elif nm in PARAM: smap[s] = sp.Symbol(f"p[{PARAM.index(nm)}]")
        elif nm == "dt":  smap[s] = sp.Symbol("dt")
        elif nm in didx:  smap[s] = sp.Symbol(f"Qaux[{didx[nm]}]")
    def body(rows):
        return "\n".join(f"    out[{i}] = {ccode(r.xreplace(smap))};" for i, r in enumerate(rows))
    layout = "\n".join(f"//   Qaux[{j}] = {nm}  ({parse(nm)[0]} d/{''.join(parse(nm)[1])})"
                       for j, nm in enumerate(derivs))
    hdr = f"""#pragma once
#include "Numerics.H"
// VAM Chorin pressure + corrector ops (generated). State: {STATE}.
// P modes = state[{ps.equation_to_state_index}]; corrector updates state[{co.equation_to_state_index}].
// Derivative-aux layout the solver fills (frozen-state h/b/q once per step;
// P-derivatives recomputed each KSP matvec):
{layout}
template <typename T>
PORTABLE_FN inline void vam_pressure_residual(const T* Q, const T* Qaux, const T* p, T dt, T* out) {{
{body(press)}
}}
template <typename T>
PORTABLE_FN inline void vam_corrector(const T* Q, const T* Qaux, const T* p, T dt, T* out) {{
{body(corr)}
}}
"""
    (out / "ChorinOps.H").write_text(hdr)
    print(f"ChorinOps.H: {len(press)} pressure rows, {len(corr)} corrector rows, {len(derivs)} derivative auxes")
    return derivs
