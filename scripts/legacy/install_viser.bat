@echo off
REM ===========================================================================
REM install_viser.bat - install the WorldGen viser fork (submodules/viser) into
REM the D-drive venv, together with its cross-platform dependencies.
REM
REM The WorldGen pyproject pins:  viser @ git+https://github.com/ZiYang-xie/viser.git
REM Since we init that fork as a submodule (submodules/viser), we install it from
REM the local checkout instead of git (GitHub direct is blocked).
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "VISER_DIR=D:\mm\WorldGen\submodules\viser"

echo [install_viser] installing viser fork from %VISER_DIR% ...
python -m pip install -e "%VISER_DIR%"
if errorlevel 1 (
    echo [install_viser] FATAL: viser install failed
    exit /b 1
)

echo [install_viser] assert import viser ...
python -c "import viser; print('viser', viser.__version__)"
if errorlevel 1 (
    echo [install_viser] FATAL: import viser failed
    exit /b 1
)

echo [install_viser] DONE
endlocal
exit /b 0
