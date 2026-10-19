@echo off
REM ===========================================================================
REM run_demo_i2s.bat - one-shot WorldGen image-to-scene (i2s) runner:
REM   inject env -> GPU gate -> demo.py -i <image> -> save splat.ply -> viser :8080
REM
REM i2s uses FLUX.1-Fill-dev + svdq-int4-fill transformer (worldgen mode='i2s'),
REM all resolved offline from the D-drive cache (Plan A).
REM
REM Usage:
REM   run_demo_i2s.bat <input_image> [prompt] [resolution] [output_dir]
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

if "%~1"=="" (
    echo usage: run_demo_i2s.bat ^<input_image^> [prompt] [resolution] [output_dir]
    exit /b 64
)
set "IMG=%~1"
set "PROMPT=%~2"
if "%PROMPT%"=="" set "PROMPT=a beautiful scene"
set "RES=%~3"
if "%RES%"=="" set "RES=1024"
set "OUT=%~4"
if "%OUT%"=="" set "OUT=D:\mm\outputs\i2s_verify"

echo [run_demo_i2s] === GPU precheck ===
powershell -ExecutionPolicy Bypass -File "D:\mm\scripts\gpu_precheck.ps1" -NeedMiB 10240
if errorlevel 1 (
    echo [run_demo_i2s] FATAL: GPU precheck failed - free VRAM before running. Aborting.
    exit /b 2
)

if not exist "%OUT%" mkdir "%OUT%"
echo [run_demo_i2s] image=%IMG% prompt="%PROMPT%" resolution=%RES% out=%OUT%
echo [run_demo_i2s] === launching demo.py (i2s) via watchdog ===
echo [run_demo_i2s] watchdog: %I2S_WATCHDOG_SECS% sec (default 300)  kill_after_artifact: %I2S_KILL_AFTER_ARTIFACT%
REM i2s has two 12GB-card defects fixed at runtime by vendor/worldgen_i2s_shim
REM (loaded by run_i2s_patched.py): ~64GiB batch_nearest_dot OOM, and the 9.5GB
REM T5 not being evicted after encode (VAE stage hangs on WDDM paging).
REM run_i2s_watchdog.py fails LOUDLY (exit 124) on a silent hang. I2S_NO_SHIM=1
REM runs the raw demo.py instead.
python "D:\mm\scripts\run_i2s_watchdog.py" -i "%IMG%" -p "%PROMPT%" -o "%OUT%" -r %RES% --low_vram --save_scene
set "RC=%errorlevel%"
echo [run_demo_i2s] demo exited with code %RC%
endlocal & exit /b %RC%
