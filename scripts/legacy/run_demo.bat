@echo off
REM ===========================================================================
REM run_demo.bat - one-shot WorldGen demo: preflight -> GPU gate -> demo.py -> collect
REM
REM   * injects the D-drive environment (set_env.bat)
REM   * asserts free VRAM >= 10 GiB (fail-fast; prints GPU-holding processes)
REM   * forces --low_vram (nunchaku SVDQuant) for the 12 GB card
REM   * launches viser at http://localhost:8080
REM
REM Usage:
REM   run_demo.bat -p "a beautiful landscape" [-r 1600] [--return_mesh] ...
REM ===========================================================================
setlocal enableextensions enabledelayedexpansion
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

REM Plan A: resolve ALL HF repos (incl. gated black-forest-labs/FLUX.1-dev) from the
REM local D-drive cache mirror -- no network, no HF auth.
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

echo [run_demo] === GPU precheck ===
powershell -ExecutionPolicy Bypass -File "D:\mm\scripts\gpu_precheck.ps1" -NeedMiB 10240
if errorlevel 1 (
    echo [run_demo] FATAL: GPU precheck failed - free VRAM before running. Aborting.
    exit /b 2
)

echo [run_demo] === building argv ===
set "ARGS="
for /f "delims=" %%a in ('python "D:\mm\scripts\demo_args.py" %*') do set "ARGS=%%a"
echo [run_demo] demo.py %ARGS%

pushd "D:\mm\WorldGen"
python demo.py %ARGS%
set "RC=%errorlevel%"
popd

echo [run_demo] === collecting outputs (exit=%RC%) ===
python "D:\mm\scripts\collect_outputs.py" --src "D:\mm\outputs\latest"

if not "%RC%"=="0" (
    echo [run_demo] WARNING: demo.py exited with code %RC%.
    exit /b %RC%
)
echo [run_demo] done. Open http://localhost:8080
endlocal
exit /b 0
