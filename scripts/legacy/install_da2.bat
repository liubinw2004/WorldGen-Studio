@echo off
REM ===========================================================================
REM install_da2.bat - install the DA-2 (360 depth, SphereViT) package.
REM   Installed with --no-deps so DA-2's torch==2.5.0 pin does NOT downgrade our
REM   torch 2.7.1 (WorldGen drives the torch version). SphereViT is pure-torch
REM   forward, no C++ extension -> expected to work on 2.7.
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "DA2=git+https://ghfast.top/https://github.com/EnVision-Research/DA-2.git#subdirectory=src"
set "UTILS3D=git+https://ghfast.top/https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"

REM utils3d is unused by WorldGen's own source, BUT da2/__init__ -> da2.utils.d2pc
REM imports it at package import time, so it IS required for `import da2`.
REM Installed --no-deps to avoid pulling utils3d's own pinned torch/numpy.
echo [install_da2] installing utils3d (--no-deps)
python -m pip install --no-deps "%UTILS3D%"
if errorlevel 1 (
    echo [install_da2] FATAL: utils3d install failed.
    exit /b 1
)

echo [install_da2] installing %DA2%  (--no-deps)
python -m pip install --no-deps "%DA2%"
if errorlevel 1 (
    echo [install_da2] FATAL: da2 install failed.
    exit /b 1
)

echo [install_da2] verifying import...
python -c "from da2.model.spherevit import SphereViT; print('da2 OK:', SphereViT)"
if errorlevel 1 (
    echo [install_da2] WARNING: import da2 failed.
)
echo [install_da2] done.
endlocal
exit /b 0
