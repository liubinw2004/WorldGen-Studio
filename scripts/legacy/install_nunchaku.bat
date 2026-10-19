@echo off
REM ===========================================================================
REM install_nunchaku.bat - fetch + install the Windows nunchaku SVDQuant wheel.
REM
REM   WorldGen pyproject:35 pins a linux_x86_64 wheel -> always fails on Windows.
REM   The matching win_amd64 wheel exists; version tag +torch2.7 aligns with our
REM   torch 2.7.1. Downloaded into D:\mm\wheels (referenced by requirements.win.txt)
REM   so it can be pinned as:  nunchaku @ file:///D:/mm/wheels/<name>.whl
REM
REM Sources (first reachable wins): hf-mirror, then ghfast proxy GitHub release.
REM The downloaded file is validated (must be a real zip/wheel, not an HTML page).
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "WHL=nunchaku-0.2.0+torch2.7-cp312-cp312-win_amd64.whl"
set "URL1=https://hf-mirror.com/mit-han-lab/nunchaku/resolve/main/%WHL%"
set "URL2=https://ghfast.top/https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/%WHL%"
set "DEST=D:\mm\wheels\%WHL%"

if not exist "D:\mm\wheels" mkdir "D:\mm\wheels"

REM Validate a candidate download: a real wheel is a zip (PK header) and lists files.
:try_download
if exist "%DEST%" (
    python -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile(r'%DEST%') else 1)" 2>nul
    if not errorlevel 1 goto :install
    echo [install_nunchaku] existing %DEST% is not a valid wheel - re-downloading.
    del /Q "%DEST%"
)

echo [install_nunchaku] downloading %WHL% via hf-mirror...
curl -L --fail --retry 3 --retry-delay 3 -o "%DEST%" "%URL1%"
if errorlevel 1 (
    echo [install_nunchaku] hf-mirror failed, trying ghfast...
    curl -L --fail --retry 3 --retry-delay 3 -o "%DEST%" "%URL2%"
)
if not exist "%DEST%" (
    echo [install_nunchaku] FATAL: could not download the wheel.
    exit /b 1
)
python -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile(r'%DEST%') else 1)" 2>nul
if errorlevel 1 (
    echo [install_nunchaku] FATAL: downloaded file is not a valid wheel - likely an HTML error page.
    exit /b 1
)

:install
echo [install_nunchaku] installing %DEST%
python -m pip install "%DEST%"
if errorlevel 1 (
    echo [install_nunchaku] FATAL: pip install nunchaku failed.
    exit /b 1
)

echo [install_nunchaku] verifying import...
python -c "import nunchaku; print('nunchaku', getattr(nunchaku,'__version__','?'), nunchaku.__file__)"
if errorlevel 1 (
    echo [install_nunchaku] WARNING: import nunchaku failed.
)
echo [install_nunchaku] done.
endlocal
exit /b 0
