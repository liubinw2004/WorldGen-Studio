@echo off
REM ===========================================================================
REM fetch_models.bat - download all tier-1 models.
REM   (1) fetch_models.py  : hf-mirror models (WorldGen LoRA, DA-2, svdq-int4)
REM   (2) assemble_flux.py : FLUX.1-dev components from ModelScope (~9.94 GB)
REM                          + Plan-A HF cache mirror
REM   Each script enforces a disk water-level guard before downloading.
REM Usage:  fetch_models.bat [--force]
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

echo [fetch_models] === tier1 non-gated (hf-mirror) ===
python "D:\mm\scripts\fetch_models.py" %*
set F1=%errorlevel%

echo [fetch_models] === FLUX.1-dev components (ModelScope) ===
python "D:\mm\scripts\assemble_flux.py" %*
set F2=%errorlevel%

echo [fetch_models] === verify ===
python "D:\mm\scripts\verify_models.py"
set F3=%errorlevel%

echo [fetch_models] exit codes: fetch=%F1% assemble=%F2% verify=%F3%
if not "%F1%%F2%%F3%"=="000" (
    echo [fetch_models] WARNING: at least one stage failed - see above.
    exit /b 1
)
echo [fetch_models] done.
endlocal
exit /b 0
