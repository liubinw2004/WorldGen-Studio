@echo off
REM ===========================================================================
REM install_deps.bat - install (a) the worldgen package itself and (b) all
REM   runtime dependencies, Windows-correctly.
REM
REM   NEVER run `pip install .` or `pip install -r pyproject` here:
REM   WorldGen/pyproject.toml:35 hard-pins a *linux_x86_64* nunchaku wheel and
REM   git+ deps -> guaranteed failure on Windows.
REM
REM   Instead:
REM     1) pip install -e D:\mm\WorldGen --no-deps   (package only, no deps)
REM     2) pip install -r D:\mm\env\requirements.win.txt
REM        (explicit pin list; nunchaku/torch point at local D:\mm\wheels files)
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "PYPI_IDX=https://mirrors.cloud.tencent.com/pypi/simple/"
set "REQS=D:\mm\env\requirements.win.txt"

if not exist "%REQS%" (
    echo [install_deps] requirements missing, regenerating via config.py --emit
    "C:\Users\Han\AppData\Local\Programs\Python\Python312\python.exe" "D:\mm\config.py" --emit
)
if not exist "D:\mm\wheels\torch-2.7.1+cu128-cp312-cp312-win_amd64.whl" (
    echo [install_deps] WARNING: torch wheel not in D:\mm\wheels - run install_torch.bat first.
)
if not exist "D:\mm\wheels\nunchaku-0.2.0+torch2.7-cp312-cp312-win_amd64.whl" (
    echo [install_deps] WARNING: nunchaku wheel not in D:\mm\wheels - run install_nunchaku.bat first.
)

echo [install_deps] step 1/2: editable install of worldgen (--no-deps)
python -m pip install --no-deps -e "D:\mm\WorldGen"
if errorlevel 1 (
    echo [install_deps] FATAL: editable install of worldgen failed.
    exit /b 1
)

echo [install_deps] step 2/2: install pinned dependencies from %REQS%
python -m pip install --index-url "%PYPI_IDX%" --trusted-host mirrors.cloud.tencent.com -r "%REQS%"
if errorlevel 1 (
    echo [install_deps] FATAL: dependency install failed.
    exit /b 1
)

echo [install_deps] freeze snapshot to D:\mm\logs\pip_freeze_deps.txt
python -m pip freeze > "D:\mm\logs\pip_freeze_deps.txt"

REM --- assertion (correction #3): nunchaku is the WINDOWS build, torch+CUDA ok,
REM     and no linux_x86_64 package sneaked into the venv. ---------------------
echo [install_deps] assertion: nunchaku/torch must be the Windows build
python -c "import nunchaku, torch; assert 'D:\\mm\\venv' in nunchaku.__file__, nunchaku.__file__; print('nunchaku:', nunchaku.__file__); print('torch:', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
if errorlevel 1 (
    echo [install_deps] FATAL: nunchaku/torch assertion failed.
    exit /b 1
)
python -c "import subprocess,sys; out=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True); bad=[l for l in out.splitlines() if 'linux_x86_64' in l.lower()]; print('linux_x86_64 packages:', bad); sys.exit(1 if bad else 0)"
if errorlevel 1 (
    echo [install_deps] FATAL: a linux_x86_64 wheel was installed into the venv.
    exit /b 1
)
echo [install_deps] done.
endlocal
exit /b 0
