#!/usr/bin/env python
"""Generate the VAM Chorin sub-model C++ headers for the dmplex backend.

VAM is hyperbolic-elliptic: VAM.chorin_split(dt) yields 3 SystemModels that
SHARE the full state [b,h,q_*,r_*,P_*]:
  * SM_pred  — hyperbolic predictor (pressure-free/hydrostatic): flux/NCP/
               eigenvalues + a bed-trace source. Lowered like any FV model.
  * SM_press — the elliptic pressure stage (affine A*P + R0); pressure_operator()
               gives analytic A0/Ax/Axx/RHS for a PETSc KSP assembly.
  * SM_corr  — pointwise corrector (update_variables).

State (verified): dimension=2 -> [b,h,q_0,q_1,r_0,r_1,P_0,P_1] (1-D horizontal
mesh, the Escalante bump); dimension=3 -> [b,h,q_x_0,q_x_1,q_y_0,q_y_1,r_0,r_1,
P_0,P_1] (2-D horizontal mesh — genuinely 2-D, q_x AND q_y).

⚠ THE SPLIT MUST BE FED A *CONFIGURED* SystemModel.
`chorin_split(dt)` with no `system_model=` builds a FRESH sm internally, so the
sub-systems inherit NO ICs and NO BCs and every slot silently falls back to the
Extrapolation default — which is what broke lake-at-rest here (a transmissive
y-boundary on the strip leaks hydrostatic pressure). Per the core docstring:
"ICs/BCs attach BEFORE the split so the sub-systems inherit them". So:

    model = VAM(..., boundary_conditions=bcs)   # BCs go in the CONSTRUCTOR
    sm    = SystemModel.from_model(model)       # (Model.system_model: REMOVED, REQ-143)
    sm.initial_conditions = ...                 # ICs on the sm, BEFORE the split
    split = model.chorin_split(dt, system_model=sm)

Verified: all three sub-systems then carry `boundary_conditions`.

The BCs mirror the WORKING reference `thesis/cases/escalante_vam_bump/run_derived.py`
(gate passes: eta RMS 8.2e-3 vs the hand-built frames). No new model, no NSM
subclass — the dmplex C++ printer emits `Model::boundary_conditions` and
TransportStep applies it, so the model's BCs are the solver's BCs.
"""
import sys, argparse
from pathlib import Path
import numpy as np
import sympy as sp
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.model.boundary_conditions import Dirichlet, Extrapolation
from zoomy_core.model.initial_conditions import Constant
from zoomy_core.systemmodel import SystemModel
from zoomy_core.numerics import ReconstructionSpec
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_c import CppModel, CppNumerics

HERE = Path(__file__).resolve().parent

Q_IN = 0.11197          # escalante_vam_bump inflow discharge


def build_bcs(dimension, inflow=True, pin_pressure=True, rotate=False):
    """Per-field BCs, mirroring run_derived.py's build_model.

    Inflow: prescribe the mean discharge on the streamwise q_0 and zero the
    higher momentum / vertical-velocity modes; h, b, P extrapolate (the default
    for any unclaimed slot). Outflow: the non-hydrostatic pressure modes are
    pinned to zero so the Chorin elliptic solve has a reference — without this
    the pressure operator is SINGULAR (run_derived.py's `pin_pressure`).

    `rotate` swaps the streamwise axis x->y for deliverable 3 (the 90°-rotated
    2-D run that tests x/y flux independence): inflow moves to `bottom`, the
    pressure pin to `top`, and q_y carries the discharge instead of q_x. The
    setup is then symmetric under the 90° rotation, so the two runs must agree
    up to the axis swap — that IS the test.

    In 2-D the CROSS-STREAM boundaries get an explicit whole-state Extrapolation
    (zero-Neumann). Two reasons this is required, not cosmetic:
      * the solution is uniform across the strip, so d/dn = 0 is EXACT there —
        it imposes nothing the extruded solution does not already satisfy;
      * every tagged face MUST resolve to a model BC. TransportStep does
        `boundary_map.at(tag_id)`, which THROWS on a tag with no BC, and a
        missing BC otherwise shows up as a step-0 nan out of the predictor.
    """
    if dimension == 2:
        stream, cross = "q", None           # 1-D horizontal: a single q family
        inlet, outlet = "left", "right"
        cross_bnds = ()                     # 1-D: no cross-stream boundary
    else:
        stream, cross = ("q_y", "q_x") if rotate else ("q_x", "q_y")
        inlet, outlet = ("bottom", "top") if rotate else ("left", "right")
        cross_bnds = ("left", "right") if rotate else ("bottom", "top")

    bcs = []
    if inflow:
        bcs += [Dirichlet(inlet, on=f"{stream}_0", value=Q_IN),
                Dirichlet(inlet, on=f"{stream}_1", value=0.0)]
        if cross is not None:               # zero the cross-stream momentum modes
            bcs += [Dirichlet(inlet, on=f"{cross}_0", value=0.0),
                    Dirichlet(inlet, on=f"{cross}_1", value=0.0)]
        bcs += [Dirichlet(inlet, on="r_0", value=0.0),
                Dirichlet(inlet, on="r_1", value=0.0)]
    if pin_pressure:
        bcs += [Dirichlet(outlet, on="P_0", value=0.0),
                Dirichlet(outlet, on="P_1", value=0.0)]
    bcs += [Extrapolation(t) for t in cross_bnds]   # on="all" -> whole state
    return bcs


def build_split(level=1, dimension=3, inflow=True, pin_pressure=True, rotate=False):
    """Declarative VAM(level, dimension) -> configured sm -> Chorin split.

    Inviscid Escalante closure: Newtonian bulk stress with nu=0 resolves the
    deviatoric moments to zero (closures=[] would leave them free symbols);
    StressFree zeros the top tangential traction.
    """
    bcs = build_bcs(dimension, inflow, pin_pressure, rotate)
    m = VAM(level=level, dimension=dimension, boundary_conditions=bcs or None,
            closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(m)
    # width-adaptive zero ICs; the real state (bed + dam-break h) is projected
    # by the solver after setup, exactly as run_derived.py does.
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    return m.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)


def emit_predictor(split, out=HERE, order=0):
    """Lower SM_pred with the STOCK NumericalSystemModel.

    No subclass: `default_operations` already gates itself on
    `_is_transport_system`, so the predictor gets the wet/dry-safe defaults (KP
    hinv desingularization + dry eigenvalue gate) while SM_press/SM_corr stay
    clean. The earlier `_UngatedNSM` (dropping the eigenvalue gate) was carried
    over from the Malpasset/SWE path and has no business here.
    """
    sm = split.SM_pred
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order),
        riemann=PositiveNonconservativeRusanov)
    # REQ-169: mark Model.H as a VAM predictor so ChorinVAM.hpp compiles its
    # real body (and its VAM_NS == n_dof_q static_assert) ONLY for a VAM tree.
    # The marker is emitted INTO Model.H — the same file that defines n_dof_q —
    # so it is rewritten by whichever generator ran last and CANNOT go stale
    # relative to the thing it guards. A -D build flag can silently disagree
    # with the checked-in headers; this cannot.
    (out / "Model.H").write_text(
        CppModel(nsm).create_code()
        + "\n// generated by create_vam_model.py (REQ-169): selects the real\n"
          "// ChorinVAMSolver. create_model.py does NOT emit this, so a non-VAM\n"
          "// tree gets the loud runtime stub instead of a compile error.\n"
          "#define ZOOMY_MODEL_IS_VAM 1\n")
    (out / "Numerics.H").write_text(
        CppNumerics(PositiveNonconservativeRusanov(model=nsm)).create_code())
    print(f"predictor: {len(sm.state)} states -> Model.H/Numerics.H (ZOOMY_MODEL_IS_VAM)")
    return nsm


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
    P_idx = list(ps.equation_to_state_index)
    C_idx = list(co.equation_to_state_index)
    # Machine-readable CA layout so the solver fills each aux slot BY SPEC
    # instead of hand-counting indices (a hand-written dim=3 table silently
    # writes P0_y into dim=2's P_1_x slot — the layout is name->index and only
    # the generator knows it). ax2 = -1 for a 1st derivative; mixed derivatives
    # (ax1 != ax2) fall out for free.
    AX = {"x": 0, "y": 1, "z": 2}
    specs = []
    for nm in derivs:
        fld, axes = parse(nm)
        ax1 = AX[axes[0]]
        ax2 = AX[axes[1]] if len(axes) > 1 else -1
        if len(axes) > 2:
            raise ValueError(f"aux {nm}: order>2 unsupported by the Deriv1 chain")
        specs.append((STATE.index(fld), ax1, ax2,
                      "true" if fld.startswith("P_") else "false", nm))
    spec_rows = "\n".join(
        f"    {{{f}, {a1}, {a2}, {isP}}},   // CA[{j}] = {nm}"
        for j, (f, a1, a2, isP, nm) in enumerate(specs))
    hdr = f"""#pragma once
#include "Numerics.H"
// VAM Chorin pressure + corrector ops (generated). State: {STATE}.
// P modes = state[{P_idx}]; corrector updates state[{C_idx}].
// Derivative-aux layout the solver fills (frozen-state h/b/q once per step;
// P-derivatives recomputed each KSP matvec):
{layout}

// ── the solver contract (GENERATED — never hard-code these) ───────────────
// These change with `dimension`: dim=2 (1-D horizontal, the Escalante bump)
// gives NF=8/NCA=10; dim=3 (2-D horizontal) gives NF=10/NCA=20. The solver
// must read them from here, or a dim=2 run silently indexes a dim=3 layout.
// NS is the PREDICTOR's evolved dof (pressure dropped by the splitter) and
// must equal Model<T>::n_dof_q — static_assert'd solver-side.
static constexpr int VAM_NF  = {len(STATE)};   // full VAM state (X + P)
static constexpr int VAM_NP  = {len(P_idx)};   // pressure modes
static constexpr int VAM_NS  = VAM_NF - VAM_NP; // evolved state dof
static constexpr int VAM_NCA = {len(derivs)};  // derivative-aux entries per cell
static constexpr int VAM_P_IDX[VAM_NP] = {{{", ".join(map(str, P_idx))}}};
static constexpr int VAM_CORR_IDX[{len(C_idx)}] = {{{", ".join(map(str, C_idx))}}};
static constexpr int VAM_N_CORR = {len(C_idx)};

// CA fill table: one row per derivative aux, in CA order.
//   field = index into the FULL VAM state (Q above)
//   ax1   = first  derivative axis (0=x, 1=y, 2=z)
//   ax2   = second derivative axis, or -1 for a 1st derivative
//   is_P  = pressure aux -> recomputed EVERY KSP matvec from the current P
//           iterate; !is_P -> frozen state aux, filled once per step.
struct VamAuxSpec {{ int field; int ax1; int ax2; bool is_P; }};
static constexpr VamAuxSpec VAM_CA[VAM_NCA] = {{
{spec_rows}
}};

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--dimension", type=int, default=3)
    ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--rotate", action="store_true",
                    help="90°-rotated 2-D setup (streamwise y) — deliverable 3")
    a = ap.parse_args()
    split = build_split(a.level, a.dimension, rotate=a.rotate)
    emit_predictor(split, order=a.order)
    emit_chorin_ops(split)
