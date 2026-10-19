@echo off
REM ===========================================================================
REM install_pytorch3d_shim.bat - register the local pytorch3d shim onto the
REM   venv's sys.path via a .pth file.
REM
REM   * the shim SOURCE stays under D:\mm\vendor\pytorch3d_shim (never copied
REM     into site-packages) so it remains auditable / diffable.
REM   * a single .pth file in site-packages adds that dir to sys.path.
REM   * NO compilation, NO MSVC, NO CUDA Toolkit.
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

echo [install_pytorch3d_shim] registering shim path via .pth
python -c "import sysconfig, pathlib; sp = pathlib.Path(sysconfig.get_paths()['purelib']); pth = sp / 'worldgen_pytorch3d_shim.pth'; pth.write_text(r'D:\mm\vendor\pytorch3d_shim', encoding='utf-8'); print('wrote', pth)"
if errorlevel 1 (
    echo [install_pytorch3d_shim] FATAL: failed to write .pth.
    exit /b 1
)

echo [install_pytorch3d_shim] verifying import matrix_to_quaternion...
python -c "import torch; from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix; import pytorch3d; print('shim', pytorch3d.__version__); print(matrix_to_quaternion(torch.eye(3).expand(4,3,3)))"
if errorlevel 1 (
    echo [install_pytorch3d_shim] FATAL: shim import failed.
    exit /b 1
)
echo [install_pytorch3d_shim] done.
endlocal
exit /b 0
