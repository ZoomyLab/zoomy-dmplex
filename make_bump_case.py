#!/usr/bin/env python
"""Build the Escalante dam-break-over-bump case for dmplex (deliverable 1).

Mirrors the WORKING reference `thesis/cases/escalante_vam_bump/run_derived.py`
(gate passes: eta RMS 8.2e-3 vs the hand-built frames) — same bump, same
dam-break IC, same domain/cells/dt, same inflow discharge.

Emits three things:
  * bump_1d.msh   — MSH 2.2, 1-D. VAM(level=1, dimension=2) lowers to
                    `Model.H dimension = 1`, so boundary_dim = dimension-1 = 0
                    and the tags must be PHYSICAL POINTS. VirtualSolver::
                    SetupArchitecture ENFORCES an exact match between
                    $PhysicalNames and Model::get_boundary_tags() (excluding
                    "default"), so these must be exactly {left, right}.
  * bump_ic.h5    — b and h per cell. The model IC is Constant(zeros) (the split
                    re-derives the predictor's signatures), and the real state is
                    projected after setup via the settings `initial_condition_file`
                    + `initial_condition_mask` — the dmplex equivalent of
                    run_derived.py assigning Q0[0]=b, Q0[1]=h after
                    setup_simulation.
  * settings_bump.json

Run (zoomy env, NOT the container — it has no python):
    python make_bump_case.py
"""
import argparse, json
from pathlib import Path
import numpy as np
import h5py

HERE = Path(__file__).resolve().parent

# ── the reference constants (escalante_vam_bump/run_derived.py) ──────────
DOMAIN = (-1.5, 1.5)
NCELLS = 60
T_END = 20.0
G = 9.81
H_RES = 0.34        # reservoir depth
H_DRY = 0.015       # dry-bed floor
Q_IN = 0.11197      # inflow discharge
DAM_X = 1.0         # reservoir occupies x < DAM_X
BUMP = lambda x: 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2))


def write_msh_1d(path, domain=DOMAIN, ncells=NCELLS):
    """MSH 2.2 1-D line mesh: `ncells` 2-node lines + 2 physical points.

    Element types: 15 = point, 1 = 2-node line. Physical dim MUST be 0 for the
    boundary tags (Model dimension=1 => boundary_dim=0); the line elements carry
    a dim-1 physical group for the cells themselves.
    """
    x = np.linspace(domain[0], domain[1], ncells + 1)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
             "$PhysicalNames", "3",
             '0 1 "left"',        # dim 0 -> boundary_dim for a 1-D model
             '0 2 "right"',
             '1 3 "domain"',
             "$EndPhysicalNames",
             "$Nodes", str(len(x))]
    for i, xi in enumerate(x, start=1):
        lines.append(f"{i} {xi:.16e} 0.0 0.0")
    lines.append("$EndNodes")
    lines.append("$Elements")
    elems, eid = [], 1
    # boundary points: `elm-type 15`, 2 tags = (physical, elementary)
    elems.append(f"{eid} 15 2 1 1 1"); eid += 1                    # left  -> node 1
    elems.append(f"{eid} 15 2 2 2 {len(x)}"); eid += 1             # right -> node N
    for c in range(ncells):                                        # cells
        elems.append(f"{eid} 1 2 3 1 {c + 1} {c + 2}"); eid += 1
    lines.append(str(len(elems)))
    lines.extend(elems)
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")
    return x


def write_ic(path, mesh_path, nstate):
    """Write the PETSc-ordered `state` Vec: flat, cell-major, block size nstate
       [Cell0_Var0, Cell0_Var1, ..., CellN_VarM]  (IOManager::LoadSolution)
    with, per run_derived.py:
        Q0[0] = b = BUMP(xc)
        Q0[1] = h = max(where(xc < DAM_X, H_RES - b, H_DRY), H_DRY)
    The settings mask [0,1] then copies ONLY those two components into X.

    ⚠ DMPlex REORDERS cells relative to the mesh file — generate_ic.py needs a
    KDTree to match meshio cells to PETSc cells for that reason. This IC is
    ANALYTIC in x, so we sidestep ordering entirely: load the plex exactly as
    the solver does and evaluate b(x)/h(x) at whatever centroid PETSc reports
    for each of ITS cells. No ordering assumption, no nearest-neighbour matching.
    """
    import petsc4py, sys as _sys
    petsc4py.init([_sys.argv[0]])
    from petsc4py import PETSc

    dm = PETSc.DMPlex().createFromFile(str(mesh_path))
    cS, cE = dm.getHeightStratum(0)          # cells
    xc = np.array([dm.computeCellGeometryFVM(c)[1][0] for c in range(cS, cE)])
    b = BUMP(xc)
    h = np.maximum(np.where(xc < DAM_X, H_RES - b, H_DRY), H_DRY)

    out = np.zeros(len(xc) * nstate, dtype=np.float64)
    out[0::nstate] = b                        # component 0 = b
    out[1::nstate] = h                        # component 1 = h
    with h5py.File(path, "w") as f:
        d = f.create_dataset("state", data=out)
        d.attrs["num_cells"] = len(xc)
        d.attrs["num_components"] = nstate
    return xc, b, h


def write_settings(path, mesh, ic, t_end=T_END, cfl=0.3):
    # dt matches run_derived.py: cfl * dx / (sqrt(g*H_RES) + 1.0)
    dx = (DOMAIN[1] - DOMAIN[0]) / NCELLS
    dt = cfl * dx / (np.sqrt(G * H_RES) + 1.0)
    cfg = {
        "name": "VAM_escalante_bump_1d",
        "io": {
            "directory": "outputs/vam_bump_1d",
            "filename": "vam_bump_1d",
            "snapshots": 40,
            "snapshot_logic": "interpolate",
            "clean_directory": True,
            "mesh_path": f"./{mesh.name}",
            "initial_condition_file": f"./{ic.name}",
            "initial_condition_mask": [0, 1],     # b, h
            "write_3d": False,
        },
        "solver": {
            "t_end": t_end,
            "cfl": cfl,
            "dt": float(dt),
            "reconstruction_order": 0,            # order-1 first; o2 after it runs
            "method": "chorin",                   # ChorinVAMSolver
            "time_integration": "splitting",
            "refresh_derivative_aux": False,
            "positivity": "mood",
        },
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncells", type=int, default=NCELLS)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--nstate", type=int, default=6,
                    help="Model<T>::n_dof_q of the generated predictor "
                         "(dim=2 -> 6: [b,h,q_0,q_1,r_0,r_1]; dim=3 -> 8)")
    a = ap.parse_args()
    NCELLS = a.ncells
    mesh = HERE / "bump_1d.msh"
    ic = HERE / "bump_ic.h5"
    cfg = HERE / "settings_bump.json"
    xn = write_msh_1d(mesh, ncells=a.ncells)
    xc, b, h = write_ic(ic, mesh, a.nstate)
    dt = write_settings(cfg, mesh, ic, t_end=a.t_end, cfl=a.cfl)
    print(f"mesh {mesh.name}: {a.ncells} cells, x in {DOMAIN}, tags left/right (dim 0)")
    print(f"ic   {ic.name}: b in [{b.min():.4f},{b.max():.4f}]  h in [{h.min():.4f},{h.max():.4f}]")
    print(f"     eta = b+h in [{(b+h).min():.4f},{(b+h).max():.4f}]  (reservoir {H_RES})")
    print(f"cfg  {cfg.name}: t_end={a.t_end}, cfl={a.cfl}, dt={dt:.5f}, ~{int(a.t_end/dt)} steps")
