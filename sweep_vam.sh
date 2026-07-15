#!/bin/bash
# Deliverable 4: dmplex VAM mesh-refinement sweep -> ns/cell/step vs N.
#
# RUN ON THE HOST, not inside the container: this needs the host micromamba
# python for make_bump_case.py (the container has no python) and shells into
# apptainer only for solver_cpu. Running the whole script inside the container
# silently produces nothing -- $ZPY does not resolve there.
#
# TWO METHODOLOGY TRAPS, both real, both measured here:
#
# (1) DO NOT BENCHMARK ONE SMALL MESH (@jax, REQ-168 thread). jax costs 2553
#     ns/cell/step at 144 cells but 241 at 409k -- a 10x kernel-launch floor. A
#     single small-mesh number measures dispatch, not the solver, and flatters
#     dmplex (CPU, in-process, no launch cost). That artefact is what made
#     REQ-122 claim "jax is ~1000x slower than foam". Hence the sweep.
#
# (2) DO NOT USE wall/steps. dmplex has a LARGE fixed startup (measured ~15 s at
#     240 cells: mesh load + DMPlex distribute + Chorin KSP/MatShell setup).
#     wall/steps charges that to the steps and overstated the per-step cost by
#     2.2x at 320 steps (84.6 vs 36.5 ms/step). So each mesh runs TWICE (t_end,
#     2*t_end) and the per-step cost is the SLOPE d(wall)/d(steps) -- the
#     startup intercept cancels exactly. The intercept is reported too; a ~15 s
#     setup at only 240 cells is worth knowing on its own.
#
# Usage:  ./sweep_vam.sh <nstate> [t_end]   e.g.  ./sweep_vam.sh 8 1.0
set -u
NSTATE=${1:?usage: sweep_vam.sh <nstate> [t_end]}
TEND=${2:-1.0}   # >=360 steps on the coarse mesh: the every-10-steps print costs <3%
ZPY=/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python
SIF=/Users/adam-obbpb5az1dhsjzf/git/ZoomySave2/zoomy_dmplex_latest.sif
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

run_one() {  # $1=nx $2=ny $3=t_end  -> echoes "steps wall_seconds"
    "$ZPY" make_bump_case.py --dim 2 --rotate --ncells "$1" --ncross "$2" \
           --nstate "$NSTATE" --t-end "$3" >/dev/null 2>&1 || { echo "0 0"; return; }
    local t0 t1 out steps
    t0=$(date +%s.%N)
    out=$(apptainer exec -B /Users -B /mnt -B /tmp "$SIF" \
              bash -lc "cd '$HERE' && mpiexec -n 1 ./solver_cpu -settings ./settings_bump_2d_rot.json" 2>&1)
    t1=$(date +%s.%N)
    steps=$(echo "$out" | grep -oE '^Step [0-9]+' | tail -1 | grep -oE '[0-9]+')
    [ -z "$steps" ] && steps=0
    echo "$steps $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')"
}

printf '%7s %5s %5s | %8s %8s | %8s %8s | %13s %10s\n' \
       cells nx ny steps_1x steps_2x wall_1x wall_2x ns/cell/step startup_s
for cfg in "60 4" "120 8" "240 16"; do
    set -- $cfg; NX=$1; NY=$2; N=$((NX * NY))
    read -r S1 W1 <<< "$(run_one "$NX" "$NY" "$TEND")"
    read -r S2 W2 <<< "$(run_one "$NX" "$NY" "$(awk -v t="$TEND" 'BEGIN{print t*2}')")"
    if [ "$S1" -gt 0 ] && [ "$S2" -gt "$S1" ]; then
        # slope d(wall)/d(steps) -> per-step cost, startup intercept removed
        NSPC=$(awk -v w1="$W1" -v w2="$W2" -v s1="$S1" -v s2="$S2" -v n="$N" \
               'BEGIN{printf "%.0f", ((w2-w1)/(s2-s1))*1e9/n}')
        ST=$(awk -v w1="$W1" -v w2="$W2" -v s1="$S1" -v s2="$S2" \
             'BEGIN{printf "%.1f", w1 - s1*(w2-w1)/(s2-s1)}')
    else
        NSPC=NA; ST=NA
    fi
    printf '%7d %5d %5d | %8d %8d | %8.1f %8.1f | %13s %10s\n' \
           "$N" "$NX" "$NY" "$S1" "$S2" "$W1" "$W2" "$NSPC" "$ST"
done
