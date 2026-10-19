#!/usr/bin/env bash
# =============================================================================
# build_viser_client.sh - build the WorldGen viser-forge client (static assets).
#
# WHY THIS SCRIPT EXISTS
#   The viser 0.2.23 runtime auto-builder (_client_autobuild._build_viser_client)
#   is BROKEN on Windows: it runs `subprocess.run([<nodeenv>/Scripts/npx, ...])`,
#   but on Windows `npx` is a POSIX shell script (the real launcher is npx.cmd),
#   so CreateProcess fails with WinError 193 ("not a valid Win32 application").
#   We therefore build the client manually with npx.cmd.
#
# PRECONDITION: node must be available (nodeenv installs Node 20.4.0 into
#   <client>/.nodeenv). This script will create it if missing.
#
# OUTPUT: <client>/build/index.html (+ assets). ViserServer() will then skip the
#   runtime build because build/index.html exists and is newer than src/.
# =============================================================================
set -eu
source /d/mm/env/env.sh

PY="/d/mm/venv/Scripts/python.exe"
CLIENT="/d/mm/WorldGen/submodules/viser/src/viser/client"

# Keep npm/yarn caches on D drive (never C).
export NPM_CONFIG_CACHE=/d/mm/cache/npm
export YARN_CACHE_FOLDER=/d/mm/cache/yarn
export npm_config_cache=/d/mm/cache/npm
mkdir -p /d/mm/cache/npm /d/mm/cache/yarn

# Ensure a sandboxed Node is present.
if [ ! -x "$CLIENT/.nodeenv/Scripts/node.exe" ]; then
    echo "[build_viser_client] installing sandboxed node 20.4.0 ..."
    "$PY" -m nodeenv --node=20.4.0 "$CLIENT/.nodeenv"
fi

export NODE_VIRTUAL_ENV="$CLIENT/.nodeenv"
export PATH="$CLIENT/.nodeenv/Scripts:$PATH"

echo "[build_viser_client] node = $(node --version)"
cd "$CLIENT"

echo "[build_viser_client] yarn install ..."
npx.cmd --yes yarn install

echo "[build_viser_client] vite build ..."
npx.cmd vite build --base ./ --outDir "$CLIENT/build"

if [ -f "$CLIENT/build/index.html" ]; then
    echo "[build_viser_client] OK -> $CLIENT/build/index.html"
else
    echo "[build_viser_client] FAILED: build/index.html missing"
    exit 1
fi
