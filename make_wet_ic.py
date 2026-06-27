#!/usr/bin/env python
"""Write a fully-wet 2-D dam-break initial condition for the dmplex solver.

The state ([b, h, q_x_0, q_y_0]) is written in PETSc DMPlex cell order (the same
ordering IOManager::LoadSolution expects), computed directly from each cell
CENTROID — so no mesh point-data / meshio round-trip is needed. The dam-break is
fully wet (h > 0 everywhere), which keeps the raw 1/h in the SME(0) flux finite;
a dry-bed case (real Malpasset bathymetry) additionally needs a desingularized
hinv flux (wet/dry closure) — see task 0025.

Run inside the PETSc container with python3.12:
    python3.12 make_wet_ic.py <mesh.msh> <out.h5> [--hL 10] [--hR 3]
"""
import sys, argparse, numpy as np, h5py
import petsc4py; petsc4py.init([])
from petsc4py import PETSc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--hL", type=float, default=10.0)
    ap.add_argument("--hR", type=float, default=3.0)
    ap.add_argument("--ncomp", type=int, default=4)
    a = ap.parse_args()

    dm = PETSc.DMPlex().createFromFile(a.mesh); dm.distribute()
    cS, cE = dm.getHeightStratum(0); ncell = cE - cS
    cx = np.array([dm.computeCellGeometryFVM(c)[1][0] for c in range(cS, cE)])
    xmid = 0.5 * (cx.min() + cx.max())

    data = np.zeros(ncell * a.ncomp)
    data[1::a.ncomp] = np.where(cx < xmid, a.hL, a.hR)   # h (b stays 0, momentum 0)
    with h5py.File(a.out, "w") as f:
        d = f.create_dataset("state", data=data)
        d.attrs["num_cells"] = ncell
        d.attrs["num_components"] = a.ncomp
    print(f"wrote {a.out}: cells={ncell} xmid={xmid:.0f} h in [{a.hR},{a.hL}]")


if __name__ == "__main__":
    main()
