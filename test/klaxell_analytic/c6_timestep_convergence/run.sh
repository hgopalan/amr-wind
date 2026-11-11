#!/bin/bash
# Usage: ./run.sh /path/to/kynema_sgf
# Runs every variant of this case in runs/<variant> and then compare.py.
set -e
EXE=${1:-${KYNEMA_SGF:?give the kynema_sgf executable as the first argument}}
HERE="$(cd "$(dirname "$0")" && pwd)"
run_case () {
    name=$1; inp=$2; shift 2
    mkdir -p "$HERE/runs/$name"
    cp "$HERE"/rans_1d.info "$HERE/runs/$name/"
    for f in "$HERE"/*.amrwind; do [ -e "$f" ] && cp "$f" "$HERE/runs/$name/"; done
    (cd "$HERE/runs/$name" && rm -rf post_processing && "$EXE" "$HERE/$inp" "$@" > log 2>&1) || echo "run $name returned non-zero"
}
run_case dt_2000 c6_timestep_convergence.inp time.fixed_dt=2 sampling.output_interval=25
run_case dt_1000 c6_timestep_convergence.inp time.fixed_dt=1 sampling.output_interval=50
run_case dt_0500 c6_timestep_convergence.inp time.fixed_dt=0.5 sampling.output_interval=100
run_case dt_0250 c6_timestep_convergence.inp time.fixed_dt=0.25 sampling.output_interval=200
python3 "$HERE/compare.py"
