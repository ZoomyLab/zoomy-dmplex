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


def write_msh_2d(path, ncells=NCELLS, ncross=4, width=0.2, rotate=False):
    """MSH 2.2 2-D structured QUAD strip: the 1-D bump extruded across.

    VAM(1, dimension=3) lowers to `Model.H dimension = 2`, so boundary_dim = 1
    and the tags must be PHYSICAL LINES (contrast the 1-D case, where
    boundary_dim = 0 and they are physical points).

    `rotate=False`: streamwise x in DOMAIN (ncells), cross-stream y in (0,width)
                    (ncross). Bump varies in x; inflow on `left`.
    `rotate=True` : the exact 90° rotation — streamwise y in DOMAIN, cross x.
                    Bump varies in y; inflow on `bottom`.
    Deliverables 2 vs 3 must agree under the axis swap; that is the whole test,
    so the two meshes are each other's transpose by construction.

    Element types: 1 = 2-node line (boundaries), 3 = 4-node quad (cells).
    """
    s = np.linspace(DOMAIN[0], DOMAIN[1], ncells + 1)     # streamwise nodes
    c = np.linspace(0.0, width, ncross + 1)               # cross-stream nodes
    # (i along streamwise, j across) -> physical (x, y)
    X, Y = (c, s) if rotate else (s, c)
    ni, nj = len(X), len(Y)
    nid = lambda i, j: j * ni + i + 1                     # 1-based, x-fastest

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
             "$PhysicalNames", "5",
             '1 1 "left"', '1 2 "right"', '1 3 "bottom"', '1 4 "top"',
             '2 5 "domain"',
             "$EndPhysicalNames", "$Nodes", str(ni * nj)]
    for j in range(nj):
        for i in range(ni):
            lines.append(f"{nid(i,j)} {X[i]:.16e} {Y[j]:.16e} 0.0")
    lines.append("$EndNodes")

    elems, eid = [], 1
    for j in range(nj - 1):                               # left / right (x=const)
        elems.append(f"{eid} 1 2 1 1 {nid(0,j)} {nid(0,j+1)}"); eid += 1
        elems.append(f"{eid} 1 2 2 2 {nid(ni-1,j)} {nid(ni-1,j+1)}"); eid += 1
    for i in range(ni - 1):                               # bottom / top (y=const)
        elems.append(f"{eid} 1 2 3 3 {nid(i,0)} {nid(i+1,0)}"); eid += 1
        elems.append(f"{eid} 1 2 4 4 {nid(i,nj-1)} {nid(i+1,nj-1)}"); eid += 1
    for j in range(nj - 1):                               # quads
        for i in range(ni - 1):
            elems.append(f"{eid} 3 2 5 1 {nid(i,j)} {nid(i+1,j)} "
                         f"{nid(i+1,j+1)} {nid(i,j+1)}"); eid += 1
    lines.append("$Elements"); lines.append(str(len(elems)))
    lines.extend(elems); lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")
    return (ni - 1) * (nj - 1)


def write_ic(path, mesh_path, nstate, rotate=False):
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
    cent = np.array([dm.computeCellGeometryFVM(c)[1] for c in range(cS, cE)])
    # the bump/dam vary along the STREAMWISE axis only (axis 1 = y when rotated)
    xc = cent[:, 1] if rotate else cent[:, 0]
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


def write_settings(path, mesh, ic, t_end=T_END, cfl=0.3, name="VAM_escalante_bump_1d", outdir="vam_bump_1d"):
    # dt matches run_derived.py: cfl * dx / (sqrt(g*H_RES) + 1.0)
    dx = (DOMAIN[1] - DOMAIN[0]) / NCELLS
    dt = cfl * dx / (np.sqrt(G * H_RES) + 1.0)
    cfg = {
        "name": name,
        "io": {
            "directory": f"outputs/{outdir}",
            "filename": outdir,
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
    ap.add_argument("--dim", type=int, default=1, choices=(1, 2),
                    help="MESH dimension: 1 = the 1-D bump (VAM dimension=2, "
                         "deliverable 1); 2 = the extruded strip (VAM "
                         "dimension=3, deliverables 2/3)")
    ap.add_argument("--rotate", action="store_true",
                    help="2-D only: the 90°-rotated setup (streamwise y) — "
                         "deliverable 3. Must match create_vam_model.py --rotate")
    ap.add_argument("--ncross", type=int, default=4, help="2-D: cells across the strip")
    ap.add_argument("--width", type=float, default=0.2, help="2-D: strip width")
    a = ap.parse_args()
    NCELLS = a.ncells

    if a.dim == 1:
        mesh, ic, cfg = HERE / "bump_1d.msh", HERE / "bump_ic.h5", HERE / "settings_bump.json"
        write_msh_1d(mesh, ncells=a.ncells)
        ncell = a.ncells
        tags = "left/right (physical POINTS, dim 0)"
        name, outdir = "VAM_escalante_bump_1d", "vam_bump_1d"
    else:
        suff = "2d_rot" if a.rotate else "2d"
        mesh = HERE / f"bump_{suff}.msh"; ic = HERE / f"bump_ic_{suff}.h5"
        cfg = HERE / f"settings_bump_{suff}.json"
        ncell = write_msh_2d(mesh, ncells=a.ncells, ncross=a.ncross,
                             width=a.width, rotate=a.rotate)
        tags = "left/right/bottom/top (physical LINES, dim 1)"
        name = f"VAM_escalante_bump_{suff}"; outdir = f"vam_bump_{suff}"

    xc, b, h = write_ic(ic, mesh, a.nstate, rotate=(a.dim == 2 and a.rotate))
    dt = write_settings(cfg, mesh, ic, t_end=a.t_end, cfl=a.cfl, name=name, outdir=outdir)
    ax = "y" if (a.dim == 2 and a.rotate) else "x"
    print(f"mesh {mesh.name}: {ncell} cells, streamwise {ax} in {DOMAIN}, tags {tags}")
    print(f"ic   {ic.name}: nstate={a.nstate}  b in [{b.min():.4f},{b.max():.4f}]  h in [{h.min():.4f},{h.max():.4f}]")
    print(f"     eta = b+h in [{(b+h).min():.4f},{(b+h).max():.4f}]  (reservoir {H_RES})")
    print(f"cfg  {cfg.name}: t_end={a.t_end}, cfl={a.cfl}, dt={dt:.5f}, ~{int(a.t_end/dt)} steps")
