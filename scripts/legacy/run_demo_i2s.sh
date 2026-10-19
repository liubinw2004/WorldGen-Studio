#!/usr/bin/env bash
# =============================================================================
# run_demo_i2s.sh - Git-Bash runner for WorldGen image-to-scene (i2s):
#   source env -> GPU gate -> demo.py -i <image> -> save splat.ply -> viser :8080
#
# i2s uses the FLUX.1-Fill-dev + svdq-int4-fill transformer path
# (WorldGen worldgen.py mode='i2s' -> build_pano_fill_model). Everything is
# resolved offline from the D-drive cache (Plan A).
#
# Usage:
#   bash /d/mm/scripts/run_demo_i2s.sh <input_image> [prompt] [resolution] [output_dir]
#
#   input_image : path to the RGB input photo (longest edge >1024 is auto-resized
#                 by worldgen.resize_img). The svdq-fill demo.jpg is a good default.
#   prompt      : text condition (default: "a beautiful scene")
#   resolution  : pano long edge; 1024 fits ~12GB VRAM, 1600 saturates it
#   output_dir  : where splat.ply is written (default: D:/mm/outputs/i2s_verify)
# =============================================================================
set -u
source /d/mm/env/env.sh

# Plan A: resolve ALL HF repos (incl. gated black-forest-labs/FLUX.1-Fill-dev)
# from the local D-drive cache mirror -- no network, no HF auth.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="/d/mm/venv/Scripts/python.exe"

if [ "$#" -lt 1 ]; then
    echo "usage: run_demo_i2s.sh <input_image> [prompt] [resolution] [output_dir]"
    exit 64
fi
IMG="$1"
PROMPT="${2:-a beautiful scene}"
RES="${3:-1024}"
OUT="${4:-D:/mm/outputs/i2s_verify}"

echo "[run_demo_i2s] === GPU precheck ==="
powershell.exe -ExecutionPolicy Bypass -File "D:\\mm\\scripts\\gpu_precheck.ps1" -NeedMiB 10240
if [ $? -ne 0 ]; then
    echo "[run_demo_i2s] FATAL: GPU precheck failed - free VRAM before running. Aborting."
    exit 2
fi

mkdir -p "$OUT"
echo "[run_demo_i2s] image=$IMG prompt='$PROMPT' resolution=$RES out=$OUT"
echo "[run_demo_i2s] === launching demo.py (i2s) via watchdog ==="
echo "[run_demo_i2s] shim: I2S_NO_SHIM=${I2S_NO_SHIM:-0}  watchdog: ${I2S_WATCHDOG_SECS:-300}s  kill_after_artifact: ${I2S_KILL_AFTER_ARTIFACT:-0}"
# NOTE: WorldGen's shipped i2s path has two defects on a 12GB card:
#   (1) map_image_to_pano -> batch_nearest_dot builds a ~64 GiB similarity
#       matrix (OOM), and
#   (2) after encode the 9.5GB T5 is never evicted, so the VAE/latent stage
#       pages on Windows WDDM and hangs silently (~120 MiB free).
# scripts/run_i2s_patched.py loads vendor/worldgen_i2s_shim which fixes BOTH at
# runtime (no source edit): chunked argmax + explicit T5 eviction after encode.
# The watchdog (run_i2s_watchdog.py) fails LOUDLY (exit 124) if a silent hang
# recurs, instead of hanging forever. Set I2S_NO_SHIM=1 to run raw demo.py.
"$PY" /d/mm/scripts/run_i2s_watchdog.py -i "$IMG" -p "$PROMPT" -o "$OUT" -r "$RES" --low_vram --save_scene
RC=$?
echo "[run_demo_i2s] demo exited with code $RC"
exit "$RC"
