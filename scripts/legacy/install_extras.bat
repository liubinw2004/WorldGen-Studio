@echo off
REM ===========================================================================
REM install_extras.bat - install the packages the source imports but pyproject
REM   forgot to declare: plyfile (splat save), scipy/imageio (demo), tqdm.
REM   Kept separate so a failure here is easy to isolate.
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "PYPI_IDX=https://mirrors.cloud.tencent.com/pypi/simple/"

echo [install_extras] installing plyfile scipy imageio tqdm
python -m pip install --index-url "%PYPI_IDX%" --trusted-host mirrors.cloud.tencent.com ^
    plyfile scipy imageio tqdm
if errorlevel 1 (
    echo [install_extras] FATAL: extras install failed.
    exit /b 1
)
echo [install_extras] done.
endlocal
exit /b 0
