@echo off
REM ===========================================================================
REM fetch_code.bat - clone ZiYang-xie/WorldGen through the GitHub proxy, with
REM   SELECTIVE submodule initialisation (per revised design).
REM
REM   * direct github.com is BLOCKED on this box (schannel TLS failure)
REM   * proxy ghfast.top is verified to clone successfully
REM   * ONLY submodules/viser and submodules/DA-2 are initialised:
REM       - submodules/pytorch3d -> replaced by vendor/pytorch3d_shim (not pulled)
REM       - submodules/ml-sharp  -> SSH URL, P2-optional  (not pulled)
REM   The clone therefore does NOT use --recursive; submodules are selected
REM   explicitly. GIT_CONFIG_GLOBAL (insteadOf rewrite) is still applied so the
REM   https submodules route through the proxy.
REM
REM Idempotent: re-running skips an already-cloned repo (use --force to reclone).
REM ===========================================================================
setlocal enableextensions
call "D:\mm\env\set_env.bat"

set "REPO_URL=https://ghfast.top/https://github.com/ZiYang-xie/WorldGen.git"
set "DEST=D:\mm\WorldGen"
set "GIT_CONFIG_GLOBAL=D:\mm\env\gitconfig"
set "GIT_TERMINAL_PROMPT=0"
set "GIT_SUBMODULES=submodules/viser submodules/DA-2"

if /I "%~1"=="--force" (
    echo [fetch_code] --force : removing %DEST%
    if exist "%DEST%" rmdir /S /Q "%DEST%"
)

if exist "%DEST%\.git" (
    echo [fetch_code] repo already present at %DEST% - skipping clone.
    goto :submodules
)

echo [fetch_code] cloning (no --recursive) %REPO_URL%
git -c "include.path=D:/mm/env/gitconfig" clone "%REPO_URL%" "%DEST%"
if errorlevel 1 (
    echo [fetch_code] FATAL: clone failed.
    exit /b 1
)

:submodules
pushd "%DEST%"
echo [fetch_code] selective submodule init: %GIT_SUBMODULES%
git -c "include.path=D:/mm/env/gitconfig" submodule update --init %GIT_SUBMODULES%
if errorlevel 1 (
    echo [fetch_code] FATAL: required submodules viser and DA-2 failed to initialise.
    popd
    exit /b 1
)

REM Ensure the two excluded submodules are NOT present (self-heal a prior
REM full-recursive clone), so the tree matches the intended contract.
for %%S in (submodules/pytorch3d submodules/ml-sharp) do (
    if exist "%%S\.git" (
        echo [fetch_code] de-initialising excluded submodule %%S
        git submodule deinit -f "%%S"
        if exist "%%S" rmdir /S /Q "%%S"
    )
)

echo [fetch_code] submodule status:
git submodule status
popd

echo [fetch_code] verifying src/worldgen presence...
if exist "%DEST%\src\worldgen\worldgen.py" (
    echo [fetch_code] OK: %DEST%\src\worldgen\worldgen.py present
) else (
    echo [fetch_code] FATAL: src/worldgen/worldgen.py missing - clone incomplete.
    exit /b 1
)
echo [fetch_code] done.
endlocal
exit /b 0
