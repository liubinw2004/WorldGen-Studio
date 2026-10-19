@echo off
REM ===========================================================================
REM build_viser_client.bat - build the WorldGen viser forge client (Windows).
REM
REM The viser 0.2.23 runtime auto-builder is broken on Windows (it execs the
REM POSIX `npx` shell script -> WinError 193). This builds the client manually
REM with npx.cmd + yarn + vite, keeping caches on D.
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"
call "D:\mm\venv\Scripts\activate.bat"

set "CLIENT=D:\mm\WorldGen\submodules\viser\src\viser\client"
set "NPM_CONFIG_CACHE=D:\mm\cache\npm"
set "YARN_CACHE_FOLDER=D:\mm\cache\yarn"
set "npm_config_cache=D:\mm\cache\npm"

if not exist "%CLIENT%\.nodeenv\Scripts\node.exe" (
    echo [build_viser_client] installing sandboxed node 20.4.0 ...
    python -m nodeenv --node=20.4.0 "%CLIENT%\.nodeenv"
)

set "NODE_VIRTUAL_ENV=%CLIENT%\.nodeenv"
set "PATH=%CLIENT%\.nodeenv\Scripts;%PATH%"

pushd "%CLIENT%"
echo [build_viser_client] yarn install ...
call npx.cmd --yes yarn install
if errorlevel 1 ( echo [build_viser_client] yarn install FAILED & popd & exit /b 1 )

echo [build_viser_client] vite build ...
call npx.cmd vite build --base ./ --outDir "%CLIENT%\build"
if errorlevel 1 ( echo [build_viser_client] vite build FAILED & popd & exit /b 1 )
popd

if exist "%CLIENT%\build\index.html" (
    echo [build_viser_client] OK
    endlocal
    exit /b 0
)
echo [build_viser_client] FAILED: build\index.html missing
endlocal
exit /b 1
