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
run_case ri_005 t4_stratified_shear.inp 'ABL.linear_profile=true' 'ABL.bottom_velocity=-80872.8543251502 0.0 0.0' 'ABL.top_velocity=-80867.6786280957 0.0 0.0' 'zlo.velocity=-2.5878485273 0.0 0.0' 'zhi.velocity=2.5878485273 0.0 0.0'
run_case ri_010 t4_stratified_shear.inp 'ABL.linear_profile=true' 'ABL.bottom_velocity=-57185.7437072255 0.0 0.0' 'ABL.top_velocity=-57182.0839367409 0.0 0.0' 'zlo.velocity=-1.8298852423 0.0 0.0' 'zhi.velocity=1.8298852423 0.0 0.0'
run_case ri_015 t4_stratified_shear.inp 'ABL.linear_profile=true' 'ABL.bottom_velocity=-46691.9642147588 0.0 0.0' 'ABL.top_velocity=-46688.9760246712 0.0 0.0' 'zlo.velocity=-1.4940950438 0.0 0.0' 'zhi.velocity=1.4940950438 0.0 0.0'
run_case ri_025 t4_stratified_shear.inp 'ABL.linear_profile=true' 'ABL.bottom_velocity=-36167.4399610947 0.0 0.0' 'ABL.top_velocity=-36165.1253190058 0.0 0.0' 'zlo.velocity=-1.1573210445 0.0 0.0' 'zhi.velocity=1.1573210445 0.0 0.0'
run_case ri_050 t4_stratified_shear.inp 'ABL.linear_profile=true' 'ABL.bottom_velocity=-25574.2420546474 0.0 0.0' 'ABL.top_velocity=-25572.6053555303 0.0 0.0' 'zlo.velocity=-0.8183495586 0.0 0.0' 'zhi.velocity=0.8183495586 0.0 0.0'
python3 "$HERE/compare.py"
