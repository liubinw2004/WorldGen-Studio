@echo off
chcp 65001 >nul
setlocal EnableExtensions
title 停止 WorldGen Studio

pushd "%~dp0"
set "ROOT=%CD%"
set "PORT=7870"

echo.
echo   正在停止 WorldGen Studio...
echo.

rem ---- 1. the studio server -------------------------------------------------
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    set "FOUND=1"
    echo   终止服务进程 PID %%P
    taskkill /F /PID %%P >nul 2>&1
)

if not defined FOUND echo   没有发现运行中的服务。

rem ---- 2. leftover inference / viewer processes (they hold VRAM) -------------
echo.
echo   检查残留的推理进程...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'demo.py|run_i2s|test_sr_pipeline' } | ForEach-Object { Write-Host ('  终止 PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

rem ---- 3. the viser viewer port ---------------------------------------------
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8080 .*LISTENING"') do (
    echo   终止查看器进程 PID %%P
    taskkill /F /PID %%P >nul 2>&1
)

echo.
echo   已清理完成。
echo.
timeout /t 2 /nobreak >nul
popd
exit /b 0
