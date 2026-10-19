#!/usr/bin/env bash
# =============================================================================
# fetch_models.sh - Git-Bash entry point for the tier-1 model download.
#   (1) fetch_models.py  : hf-mirror models (WorldGen LoRA, DA-2, svdq-int4)
#   (2) assemble_flux.py : FLUX.1-dev components from ModelScope (~9.94 GB)
#   (3) verify_models.py : completeness check
# Usage:  bash /d/mm/scripts/fetch_models.sh [--force]
# =============================================================================
set -u
source /d/mm/env/env.sh

PY="/d/mm/venv/Scripts/python.exe"
STAGE1=0; STAGE2=0; STAGE3=0

echo "[fetch_models] === tier1 non-gated (hf-mirror) ==="
"$PY" /d/mm/scripts/fetch_models.py "$@" || STAGE1=$?

echo "[fetch_models] === FLUX.1-dev components (ModelScope) ==="
"$PY" /d/mm/scripts/assemble_flux.py "$@" || STAGE2=$?

echo "[fetch_models] === verify ==="
"$PY" /d/mm/scripts/verify_models.py || STAGE3=$?

echo "[fetch_models] exit codes: fetch=$STAGE1 assemble=$STAGE2 verify=$STAGE3"
if [ "$STAGE1" -ne 0 ] || [ "$STAGE2" -ne 0 ] || [ "$STAGE3" -ne 0 ]; then
    echo "[fetch_models] WARNING: at least one stage failed - see above."
    exit 1
fi
echo "[fetch_models] done."
