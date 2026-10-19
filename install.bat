@echo off
rem ===========================================================================
rem  WorldGen Studio - installer
rem
rem  Usage:
rem    install.bat            normal install (official sources)
rem    install.bat --mirror   mainland-China networks (hf-mirror + GitHub proxy)
rem
rem  This script only finds Python, creates the venv, and hands over to
rem  scripts\bootstrap.py. Model weights are NOT downloaded here -- they are
rem  large and partly gated; see README "Model setup".
rem
rem  NOTE: this file is deliberately ASCII-only. cmd.exe mis-parses UTF-8
rem  Chinese text in .bat files, which makes scripts fail with bogus
rem  "'...' is not recognized" errors.
rem ===========================================================================
setlocal EnableExtensions
title WorldGen Studio - installer

pushd "%~dp0"
set "ROOT=%CD%"

echo.
echo   WorldGen Studio - installer
echo   ---------------------------------------------
echo   Project root: %ROOT%
echo.

set "MIRROR_ARG="
if /I "%~1"=="--mirror" set "MIRROR_ARG=--mirror"
if defined MIRROR_ARG (
    echo   Mode: China mirrors
) else (
    echo   Mode: official sources  ^(use install.bat --mirror in mainland China^)
)
echo.

rem ---- 1. locate Python 3.11/3.12 -------------------------------------------
echo   [1/4] Looking for Python 3.11 / 3.12 ...
set "BASE_PY="

py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "BASE_PY=py -3.12"

if not defined BASE_PY (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "BASE_PY=py -3.11"
)

if not defined BASE_PY (
    for /f "tokens=2" %%v in ('python -V 2^>^&1') do (
        echo %%v | findstr /B /C:"3.12" >nul && set "BASE_PY=python"
        echo %%v | findstr /B /C:"3.11" >nul && set "BASE_PY=python"
    )
)

if not defined BASE_PY (
    echo.
    echo   [ERROR] Python 3.11 or 3.12 not found.
    echo.
    echo   Install Python 3.12 from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" during setup.
    echo   Python 3.13 will NOT work: torch cu128 and nunchaku only ship
    echo   cp312 wheels.
    echo.
    pause
    popd
    exit /b 1
)

for /f "delims=" %%v in ('%BASE_PY% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%v"
echo         Using Python %PYVER%  ^(%BASE_PY%^)

rem ---- 2. generate env config ----------------------------------------------
echo   [2/4] Generating environment config ...
%BASE_PY% "%ROOT%\config.py" --emit >nul
if errorlevel 1 (
    echo   [ERROR] config.py --emit failed.
    pause
    popd
    exit /b 1
)
call "%ROOT%\env\set_env.bat"
set "PIP_CONFIG_FILE=%ROOT%\env\pip.ini"
if defined MIRROR_ARG set "WORLDGEN_MIRROR=1"

rem ---- 3. create the venv --------------------------------------------------
echo   [3/4] Preparing virtual environment ...
if exist "%ROOT%\venv\Scripts\python.exe" (
    echo         venv already exists, skipping
) else (
    %BASE_PY% -m venv "%ROOT%\venv"
    if errorlevel 1 (
        echo   [ERROR] Failed to create the virtual environment.
        pause
        popd
        exit /b 1
    )
    echo         venv created
)

"%ROOT%\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
if errorlevel 1 echo   [WARN] pip upgrade failed, continuing anyway.

rem ---- 4. install dependencies ---------------------------------------------
echo   [4/4] Installing dependencies (10-20 min on first run) ...
echo.
"%ROOT%\venv\Scripts\python.exe" "%ROOT%\scripts\bootstrap.py" %MIRROR_ARG%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   Done. Place the model files listed above, then run start.bat.
) else (
    echo   Install did not finish. See the errors above.
    echo   Fix the cause and re-run this script: finished steps are skipped.
)
echo.
pause
popd
exit /b %RC%
