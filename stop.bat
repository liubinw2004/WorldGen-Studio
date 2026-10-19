@echo off
rem ===========================================================================
rem  WorldGen Studio - stop the server and any leftover inference processes.
rem  Leftover viewers keep several GB of VRAM alive, which makes the next run
rem  fail with "not enough VRAM".
rem
rem  NOTE: ASCII-only on purpose -- cmd.exe mis-parses UTF-8 Chinese in .bat.
rem ===========================================================================
setlocal EnableExtensions
title WorldGen Studio - stop

pushd "%~dp0"
set "ROOT=%CD%"
set "PORT=7870"

echo.
echo   Stopping WorldGen Studio ...
echo.

rem ---- 1. the studio server -------------------------------------------------
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    set "FOUND=1"
    echo   Killing server PID %%P
    taskkill /F /PID %%P >nul 2>&1
)
if not defined FOUND echo   No server was running.

rem ---- 2. leftover inference / viewer processes (they hold VRAM) ------------
echo.
echo   Checking for leftover inference processes ...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'demo.py|run_i2s|test_sr_pipeline' } | ForEach-Object { Write-Host ('  killing PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

rem ---- 3. the viser viewer port --------------------------------------------
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8080 .*LISTENING"') do (
    echo   Killing viewer PID %%P
    taskkill /F /PID %%P >nul 2>&1
)

echo.
echo   Done.
echo.
timeout /t 2 /nobreak >nul
popd
exit /b 0
