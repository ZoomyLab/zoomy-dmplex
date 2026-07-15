#!/bin/bash
# REQ-172: is FIELDSPLIT/Schur MESH-INDEPENDENT, and does it clear the dx=0.0125
# wall where PCNONE gives 246/246 DIVERGED_ITS?
#
# @amrex's point is the whole reason this exists: 60 cells was NEVER BROKEN
# (PCNONE converges 58/58 there), so Schur winning at 60 cells proves nothing.
# The deciding test is whether the iteration count STAYS flat as dx shrinks.
# amrex is gating their own roadmap (structured coupling from the emitted
# A0/Ax/Axx blocks instead of an MLMG Vanka) on this curve.
#
# MEASUREMENT PROTOCOL -- both halves matter and both bit me once:
#  * Count the OUTER pressure KSP only, via ZOOMY_VAM_DIAG ("reason=N its=M",
#    from KSPGetIterationNumber). Do NOT use -ksp_converged_reason: under
#    fieldsplit it reports the inner sub-KSPs alongside the outer and the mix
#    silently understates the count.
#  * Same t_end for every row, or the solve counts are not comparable (a
#    t_end-mismatched baseline nearly gave me a bogus 38x).
# PETSc reason: 2 = CONVERGED_RTOL, 3 = CONVERGED_ATOL, negative = DIVERGED.
#
# Run ON THE HOST (needs the host python for make_bump_case.py).
set -u
TEND=${1:-0.05}
ZPY=/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python
SIF=/Users/adam-obbpb5az1dhsjzf/git/ZoomySave2/zoomy_dmplex_latest.sif
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
FS="-pc_type fieldsplit -pc_fieldsplit_block_size 2 -pc_fieldsplit_0_fields 0 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur"

tally() {  # stdin -> "reason:count(mean its) ..."
    grep -oE "reason=[-0-9]+ its=[0-9]+" \
    | awk -F'[= ]' '{r=$2;n=$4;c[r]++;s[r]+=n}
        END{for(k in c) printf "r%s:%d(%.1f) ", k, c[k], s[k]/c[k]}'
}

printf '%6s %9s | %-34s | %-34s\n' N dx "PCNONE  (outer KSP)" "SCHUR   (outer KSP)"
for N in 60 240 960 3840; do
    DX=$(awk -v n="$N" 'BEGIN{printf "%.6f", 3.0/n}')
    "$ZPY" make_bump_case.py --dim 1 --ncells "$N" --nstate 6 --t-end "$TEND" >/dev/null 2>&1 \
        || { printf '%6d %9s | gen FAILED\n' "$N" "$DX"; continue; }
    A=$(timeout 900 apptainer exec -B /Users -B /mnt "$SIF" bash -lc \
         "cd '$HERE' && ZOOMY_VAM_DIAG=1 ZOOMY_VAM_PCNONE=1 mpiexec -n 1 ./solver_cpu -settings ./settings_bump.json" 2>/dev/null | tally)
    B=$(timeout 900 apptainer exec -B /Users -B /mnt "$SIF" bash -lc \
         "cd '$HERE' && ZOOMY_VAM_DIAG=1 mpiexec -n 1 ./solver_cpu -settings ./settings_bump.json $FS" 2>/dev/null | tally)
    printf '%6d %9s | %-34s | %-34s\n' "$N" "$DX" "${A:-<none>}" "${B:-<none>}"
done
