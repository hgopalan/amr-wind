#!/bin/bash
# Usage: ./run_all.sh /path/to/kynema_sgf [jobs]
# Runs every case (solver runs and comparisons), then prints the summary table.
set -e
EXE=${1:-${KYNEMA_SGF:?give the kynema_sgf executable as the first argument}}
JOBS=${2:-4}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
ls -d t?_* c?_* | xargs -P "$JOBS" -I{} sh -c "./{}/run.sh '$EXE' > {}/run.log 2>&1 || true"
python3 "$HERE/summarize.py"
