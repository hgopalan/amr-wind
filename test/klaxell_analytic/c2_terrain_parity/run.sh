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
run_case neutral_flat c2_neutral.inp
run_case neutral_terrain c2_neutral.inp 'incflo.physics=ABL TerrainDrag' TerrainDrag.terrain_file=terrain.amrwind
run_case stratified_flat c2_stratified.inp
run_case stratified_terrain c2_stratified.inp 'incflo.physics=ABL TerrainDrag' TerrainDrag.terrain_file=terrain.amrwind
python3 "$HERE/compare.py"
