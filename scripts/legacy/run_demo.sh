#!/usr/bin/env bash
# =============================================================================
# run_demo.sh - Git-Bash WorldGen demo runner:
#   source env -> GPU gate -> build argv -> demo.py -> collect outputs
#
# Usage:  bash /d/mm/scripts/run_demo.sh -p "a beautiful landscape" [-r 1600]
# =============================================================================
set -u
source /d/mm/env/env.sh

# Plan A: resolve ALL HF repos (incl. gated black-forest-labs/FLUX.1-dev) from the
# local D-drive cache mirror -- no network, no HF auth.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="/d/mm/venv/Scripts/python.exe"

echo "[run_demo] === GPU precheck ==="
# git-bash -> powershell bridge
powershell.exe -ExecutionPolicy Bypass -File "D:\\mm\\scripts\\gpu_precheck.ps1" -NeedMiB 10240
if [ $? -ne 0 ]; then
    echo "[run_demo] FATAL: GPU precheck failed - free VRAM before running. Aborting."
    exit 2
fi

echo "[run_demo] === building argv ==="
ARGS="$("$PY" /d/mm/scripts/demo_args.py "$@")"
echo "[run_demo] demo.py $ARGS"

pushd /d/mm/WorldGen >/dev/null
# shellcheck disable=SC2086
$PY demo.py $ARGS
RC=$?
popd >/dev/null

echo "[run_demo] === collecting outputs (exit=$RC) ==="
"$PY" /d/mm/scripts/collect_outputs.py --src "D:\\mm\\outputs\\latest"

if [ "$RC" -ne 0 ]; then
    echo "[run_demo] WARNING: demo.py exited with code $RC."
    exit "$RC"
fi
echo "[run_demo] done. Open http://localhost:8080"
