@echo off
REM ===========================================================================
REM create_venv.bat - create the D-drive venv using the SYSTEM Python 3.12.7.
REM
REM The system interpreter (C:\...\Python312\python.exe) is the ONLY permitted
REM C:-drive reference and is used read-only purely as the venv creator. The venv
REM body itself lives entirely on D:. No C:-drive bytes are produced.
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"

set "SYS_PY=C:\Users\Han\AppData\Local\Programs\Python\Python312\python.exe"
set "VENV=D:\mm\venv"

if not exist "%SYS_PY%" (
    echo [create_venv] FATAL: system python not found: %SYS_PY%
    exit /b 1
)

if exist "%VENV%\Scripts\python.exe" (
    echo [create_venv] venv already exists at %VENV% - skipping.
) else (
    echo [create_venv] creating venv at %VENV% from %SYS_PY%
    "%SYS_PY%" -m venv "%VENV%"
    if errorlevel 1 (
        echo [create_venv] FATAL: venv creation failed.
        exit /b 1
    )
)

echo [create_venv] upgrading pip/setuptools/wheel (cache redirected to D:)...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [create_venv] WARNING: pip upgrade failed (continuing).
)

echo [create_venv] interpreter info:
"%VENV%\Scripts\python.exe" -c "import sys; print(sys.version); print(sys.executable)"
echo [create_venv] done.
endlocal
exit /b 0
