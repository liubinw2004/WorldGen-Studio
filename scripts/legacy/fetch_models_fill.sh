#!/usr/bin/env bash
# =============================================================================
# fetch_models_fill.sh - Git-Bash one-shot fetch for the i2s (image-to-scene)
# capability:
#   (1) assemble_flux.py --variant fill : FLUX.1-Fill-dev components from
#       ModelScope (~0.6 GB new; T5/CLIP-L hardlink-reused from FLUX.1-dev after
#       sha256 verification) + Plan-A HF cache mirror
#   (2) fetch_models.py --tier 2        : svdq-int4-flux.1-fill-dev 4-bit
#       nunchaku fill transformer (+ demo.jpg test input image)
#
# Idempotent: already-present files with the correct size are skipped.
# Usage:  bash /d/mm/scripts/fetch_models_fill.sh [--force]
# =============================================================================
set -u
source /d/mm/env/env.sh

PY="/d/mm/venv/Scripts/python.exe"
STAGE1=0; STAGE2=0

echo "[fetch_models_fill] === FLUX.1-Fill-dev components (ModelScope) ==="
"$PY" /d/mm/scripts/assemble_flux.py --variant fill "$@" || STAGE1=$?

echo "[fetch_models_fill] === tier2 fill transformer (hf-mirror) ==="
"$PY" /d/mm/scripts/fetch_models.py --tier 2 "$@" || STAGE2=$?

echo "[fetch_models_fill] exit codes: assemble_fill=$STAGE1 fetch_tier2=$STAGE2"
if [ "$STAGE1" -ne 0 ] || [ "$STAGE2" -ne 0 ]; then
    echo "[fetch_models_fill] WARNING: at least one stage failed - see above."
    exit 1
fi
echo "[fetch_models_fill] done."
