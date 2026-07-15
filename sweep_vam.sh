#!/bin/bash
# Deliverable 4: dmplex VAM mesh-refinement sweep -> ns/cell/step vs N.
#
# WHY A SWEEP AND NOT ONE NUMBER (@jax, REQ-168 thread): jax costs 2553
# ns/cell/step at 144 cells but 241 at 409k -- a 10x kernel-launch floor. A
# single small-mesh number measures dispatch overhead, not the solver, and it
# would flatter dmplex (CPU, in-process, no launch cost). That artefact is
# exactly what made REQ-122 claim "jax is ~1000x slower than foam".
#
# Fixed t_end per run; dt is CFL-adaptive so finer meshes take proportionally
# more steps. ns/cell/step normalises both out.
#
# Usage:  ./sweep_vam.sh <nstate> [t_end]
set -u
NSTATE=${1:?usage: sweep_vam.sh <nstate> [t_end]}
TEND=${2:-1.0}   # >=360 steps on the coarse mesh: the every-10-steps print costs <3%
ZPY=/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

printf '%8s %8s %8s %10s %12s %14s\n' cells nx ny steps wall_s ns/cell/step
for cfg in "60 4" "120 8" "240 16"; do
    set -- $cfg; NX=$1; NY=$2
    N=$((NX * NY))
    $ZPY make_bump_case.py --dim 2 --rotate --ncells "$NX" --ncross "$NY" \
         --nstate "$NSTATE" --t-end "$TEND" > /dev/null 2>&1 || { echo "gen failed $NX x $NY"; continue; }
    T0=$(date +%s.%N)
    OUT=$(mpiexec -n 1 ./solver_cpu -settings ./settings_bump_2d_rot.json 2>&1 | grep -v Authorization)
    T1=$(date +%s.%N)
    STEPS=$(echo "$OUT" | grep -oE '^Step [0-9]+' | tail -1 | grep -oE '[0-9]+')
    [ -z "$STEPS" ] && STEPS=0
    WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')
    if [ "$STEPS" -gt 0 ]; then
        NSPC=$(awk -v w="$WALL" -v s="$STEPS" -v n="$N" 'BEGIN{printf "%.0f", w*1e9/(s*n)}')
    else
        NSPC=NA
    fi
    printf '%8d %8d %8d %10d %12.2f %14s\n' "$N" "$NX" "$NY" "$STEPS" "$WALL" "$NSPC"
done
