#!/usr/bin/env bash
# =============================================================================
# run_bootstrap.sh - idempotent, end-to-end WorldGen-on-Windows reproduction.
#
# Every step is guarded by state.json (D:\mm\state.json): a step whose key is
# already "done" is SKIPPED. Re-running the bootstrap after a success therefore
# does almost nothing (fast no-op). Use --force to re-run steps anyway.
#
# Pipeline
#   T01  config --emit (env files + dir tree) , preflight (C-drive baseline)
#   T02  fetch_code -> create_venv -> torch -> deps -> nunchaku -> da2 -> extras
#        -> pytorch3d shim -> smoke_test
#   T03  fetch_models (tier1 hf-mirror) -> assemble_flux (ModelScope) -> verify_models
#   T03b install_viser -> build_viser_client
#   T04  (opt-in, needs GPU) demo background -> wait for splat.ply -> collect
#   T05  verify_c_drive -> validate_artifacts
#
# Usage:
#   bash run_bootstrap.sh                 # T01..T03b + T05 (no demo)
#   bash run_bootstrap.sh --with-demo     # also run T04 (blocking-ish, ~5 min)
#   bash run_bootstrap.sh --from T03 --to T05
#   bash run_bootstrap.sh --list          # show step status only
#   bash run_bootstrap.sh --force T02.smoke_test
# =============================================================================
set -u
source /d/mm/env/env.sh

PY="/d/mm/venv/Scripts/python.exe"
SRC="/d/mm/scripts"
LOG="/d/mm/logs"
PS="powershell.exe"
mkdir -p "$LOG"

FROM="T01"; TO="T05"; LIST_ONLY=0; WITH_DEMO=0; FORCE=""
DONE_KEYS=""; DONE_KEYS_NL=$'\n'
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2;;
        --to)   TO="$2"; shift 2;;
        --list) LIST_ONLY=1; shift;;
        --with-demo) WITH_DEMO=1; shift;;
        --force) FORCE="$2"; shift 2;;
        *) echo "unknown arg: $1"; exit 64;;
    esac
done

is_done() {
    # pure-bash membership test (no subprocess spawn)
    case "${DONE_KEYS_NL}" in
        *$'\n'"$1"$'\n'*) return 0 ;;
        *) return 1 ;;
    esac
}
reload_done_keys() {
    # NB: python on Windows prints CRLF -> strip CR or the \n-delimited match fails.
    DONE_KEYS=$("$PY" -c "import sys;sys.path.insert(0,'/d/mm');import config as cfg;print('\n'.join(k for k,v in cfg.load_state().get('steps',{}).items() if v.get('done')))" 2>/dev/null | tr -d '\r')
    DONE_KEYS_NL=$'\n'"$DONE_KEYS"$'\n'
}
mark() {
    "$PY" -c "import sys;sys.path.insert(0,'/d/mm');import config as cfg;cfg.mark_step('$1',{'by':'run_bootstrap'})" 2>/dev/null
    DONE_KEYS_NL="${DONE_KEYS_NL}${1}"$'\n'
}

# step: <rank> <state_key> <desc> <cmd...>
run_step() {
    local key="$1"; local desc="$2"; shift 2
    local rank="${key%%[.]*}"
    # range filter
    if [ "$rank" \< "$FROM" ] || [ "$rank" \> "$TO" ]; then return 0; fi
    if [ "$LIST_ONLY" = "1" ]; then
        if is_done "$key"; then echo "  [done]   $key  ($desc)"; else echo "  [PENDING] $key  ($desc)"; fi
        return 0
    fi
    if is_done "$key" && [ "$FORCE" != "$key" ]; then
        echo "== SKIP  $key  (already done)"
        return 0
    fi
    echo "== RUN   $key  ($desc)"
    if "$@"; then
        mark "$key"
        echo "== OK    $key"
        return 0
    fi
    echo "== FAIL  $key  (see output above; keep state unmarked so a rerun retries)"
    return 1
}

# ---- T01 --------------------------------------------------------------------
step_t01_emit()  { "$PY" /d/mm/config.py --emit; }
step_t01_preflight() {
    if [ -f "$LOG/c_drive_baseline.json" ]; then
        echo "  baseline exists, keeping it (delete to re-snapshot)"; return 0
    fi
    "$PS" -ExecutionPolicy Bypass -File "D:\\mm\\scripts\\preflight.ps1"
}

# ---- T02 --------------------------------------------------------------------
step_fetch_code()    { cmd //c "D:\\mm\\scripts\\fetch_code.bat"; }
step_create_venv()   { cmd //c "D:\\mm\\scripts\\create_venv.bat"; }
step_install_torch() { cmd //c "D:\\mm\\scripts\\install_torch.bat"; }
step_install_deps()  { cmd //c "D:\\mm\\scripts\\install_deps.bat"; }
step_install_nunchaku() { cmd //c "D:\\mm\\scripts\\install_nunchaku.bat"; }
step_install_da2()   { cmd //c "D:\\mm\\scripts\\install_da2.bat"; }
step_install_extras(){ cmd //c "D:\\mm\\scripts\\install_extras.bat"; }
step_install_shim()  { cmd //c "D:\\mm\\scripts\\install_pytorch3d_shim.bat"; }
step_smoke()         { "$PY" "$SRC/smoke_test.py"; }

# ---- T03 --------------------------------------------------------------------
step_fetch_models()  { "$PY" "$SRC/fetch_models.py"; }
step_assemble_flux() { "$PY" "$SRC/assemble_flux.py"; }
step_verify_models() { "$PY" "$SRC/verify_models.py"; }

# ---- T03b (viser) -----------------------------------------------------------
step_install_viser()  { cmd //c "D:\\mm\\scripts\\install_viser.bat"; }
step_build_viser()    { bash "$SRC/build_viser_client.sh"; }

# ---- T04 (opt-in) -----------------------------------------------------------
step_demo() {
    echo "  launching demo in background ..."
    ( cd /d/mm/WorldGen && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" demo.py \
        -p "a beautiful tropical beach with palm trees and turquoise water, sunny sky, HDR" \
        -o "D:/mm/outputs/latest" -r 1600 --low_vram --save_scene \
        > "$LOG/demo.console.log" 2>&1 ) &
    local pid=$!
    local waited=0
    while [ $waited -lt 600 ]; do
        if [ -s /d/mm/outputs/latest/splat.ply ]; then
            echo "  splat.ply produced after ${waited}s"
            sleep 3
            # stop the blocking viser server by the PID listening on :8080
            local wpid
            wpid=$(netstat -ano 2>/dev/null | grep ':8080' | grep -i LISTENING | awk '{print $NF}' | head -1)
            if [ -n "${wpid:-}" ]; then taskkill //F //PID "$wpid" >/dev/null 2>&1 && echo "  stopped viser pid=$wpid"; fi
            kill "$pid" 2>/dev/null
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  demo process exited before producing splat.ply"; return 1
        fi
        sleep 5; waited=$((waited+5))
    done
    echo "  timeout waiting for splat.ply"; kill "$pid" 2>/dev/null; return 1
}
step_collect() { "$PY" "$SRC/collect_outputs.py" --src "D:/mm/outputs/latest"; }

# ---- T05 --------------------------------------------------------------------
step_verify_cdrive() { "$PS" -ExecutionPolicy Bypass -File "D:\\mm\\scripts\\verify_c_drive.ps1"; }
step_validate()      { "$PY" "$SRC/validate_artifacts.py"; }

echo "########## WorldGen bootstrap (from=$FROM to=$TO demo=$WITH_DEMO list=$LIST_ONLY) ##########"
reload_done_keys

rc=0
run_step "T01"            "emit env files + dir tree"            step_t01_emit            || rc=1
run_step "T01.preflight"  "C-drive baseline snapshot"            step_t01_preflight       || rc=1
run_step "T02.fetch_code" "clone WorldGen via proxy"             step_fetch_code          || rc=1
run_step "T02.create_venv" "create D-drive venv"                step_create_venv         || rc=1
run_step "T02.install_torch" "torch+torchvision cu128"          step_install_torch       || rc=1
run_step "T02.install_deps" "worldgen + requirements.win"       step_install_deps        || rc=1
run_step "T02.install_nunchaku" "nunchaku win wheel"            step_install_nunchaku    || rc=1
run_step "T02.install_da2"  "DA-2 + utils3d"                    step_install_da2         || rc=1
run_step "T02.install_extras" "plyfile/scipy/..."               step_install_extras      || rc=1
run_step "T02.install_pytorch3d_shim" "pytorch3d shim"          step_install_shim        || rc=1
run_step "T02.smoke_test" "6-rule smoke test"                   step_smoke               || rc=1
run_step "T03.fetch_models" "tier1 models (hf-mirror)"          step_fetch_models        || rc=1
run_step "T03.assemble_flux" "FLUX.1-dev components (ModelScope)" step_assemble_flux     || rc=1
run_step "T03.verify_models" "model completeness"               step_verify_models       || rc=1
run_step "T03b.install_viser" "install viser fork"              step_install_viser       || rc=1
run_step "T03b.build_viser" "build viser client"                step_build_viser         || rc=1
if [ "$WITH_DEMO" = "1" ]; then
    run_step "T04.demo" "run text-to-scene demo"                step_demo                || rc=1
    run_step "T04.collect" "collect artifacts"                  step_collect             || rc=1
fi
run_step "T05.verify_cdrive" "C-drive write check"             step_verify_cdrive       || rc=1
run_step "T05.validate" "consolidated validation"              step_validate            || rc=1

echo "########## bootstrap finished (rc=$rc) ##########"
exit $rc
