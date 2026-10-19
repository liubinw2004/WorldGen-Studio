@echo off
REM ===========================================================================
REM fetch_models_fill.bat - one-shot fetch for the i2s (image-to-scene)
REM capability:
REM   (1) assemble_flux.py --variant fill : FLUX.1-Fill-dev components from
REM       ModelScope (~0.6 GB new; T5/CLIP-L hardlink-reused from FLUX.1-dev
REM       after sha256 verification) + Plan-A HF cache mirror
REM   (2) fetch_models.py --tier 2        : svdq-int4-flux.1-fill-dev 4-bit
REM       nunchaku fill transformer (+ demo.jpg test input image)
REM
REM Idempotent: already-present files with the correct size are skipped.
REM Usage:  fetch_models_fill.bat [--force]
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

echo [fetch_models_fill] === FLUX.1-Fill-dev components (ModelScope) ===
python "D:\mm\scripts\assemble_flux.py" --variant fill %*
set F1=%errorlevel%

echo [fetch_models_fill] === tier2 fill transformer (hf-mirror) ===
python "D:\mm\scripts\fetch_models.py" --tier 2 %*
set F2=%errorlevel%

echo [fetch_models_fill] exit codes: assemble_fill=%F1% fetch_tier2=%F2%
if not "%F1%%F2%"=="00" (
    echo [fetch_models_fill] WARNING: at least one stage failed - see above.
    exit /b 1
)
echo [fetch_models_fill] done.
endlocal
exit /b 0
