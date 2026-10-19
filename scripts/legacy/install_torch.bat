@echo off
REM ===========================================================================
REM install_torch.bat - fetch + install torch 2.7.1+cu128 & torchvision 0.22.1+cu128
REM   (cp312 win_amd64) for the D-drive venv.
REM
REM SOURCE DECISION (measured on this box):
REM   * aliyun pytorch-wheels/cu128/ -> HTML landing page (not PEP503) AND
REM     throttled to ~0.43 MB/s  => unusable for a 3 GB wheel.
REM   * download.pytorch.org/whl/cu128 -> reachable, ~7.8 MB/s, HTTP range OK
REM     => chosen. Wheels are fetched with curl (resumable) into D:\mm\wheels,
REM     then installed from the LOCAL files so no unpack temp lands on C:.
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "TORCH_WHL_URL=https://download.pytorch.org/whl/cu128/torch-2.7.1%%2Bcu128-cp312-cp312-win_amd64.whl"
set "TV_WHL_URL=https://download.pytorch.org/whl/cu128/torchvision-0.22.1%%2Bcu128-cp312-cp312-win_amd64.whl"
set "TORCH_WHL=D:\mm\wheels\torch-2.7.1+cu128-cp312-cp312-win_amd64.whl"
set "TV_WHL=D:\mm\wheels\torchvision-0.22.1+cu128-cp312-cp312-win_amd64.whl"
set "PYPI_IDX=https://mirrors.cloud.tencent.com/pypi/simple/"

if not exist "D:\mm\wheels" mkdir "D:\mm\wheels"

echo [install_torch] downloading torch wheel (resumable) to %TORCH_WHL%
if not exist "%TORCH_WHL%" (
    curl -L --fail --retry 5 --retry-delay 3 -C - -o "%TORCH_WHL%" "%TORCH_WHL_URL%"
    if errorlevel 1 ( echo [install_torch] FATAL: torch wheel download failed. & exit /b 1 )
)
echo [install_torch] downloading torchvision wheel to %TV_WHL%
if not exist "%TV_WHL%" (
    curl -L --fail --retry 5 --retry-delay 3 -C - -o "%TV_WHL%" "%TV_WHL_URL%"
    if errorlevel 1 ( echo [install_torch] FATAL: torchvision wheel download failed. & exit /b 1 )
)

echo [install_torch] installing local wheels + deps from %PYPI_IDX%
python -m pip install ^
    --index-url "%PYPI_IDX%" ^
    --trusted-host mirrors.cloud.tencent.com ^
    "%TORCH_WHL%" "%TV_WHL%"
if errorlevel 1 (
    echo [install_torch] FATAL: torch install failed.
    exit /b 1
)

echo [install_torch] verifying CUDA availability...
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if errorlevel 1 (
    echo [install_torch] WARNING: torch import check failed.
)
echo [install_torch] done.
endlocal
exit /b 0
