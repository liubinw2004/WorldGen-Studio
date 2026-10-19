@echo off
chcp 65001 >nul
setlocal EnableExtensions
title WorldGen Studio

rem ---------------------------------------------------------------------------
rem  Start the local server and open the app window.
rem  Works from any directory: ROOT is resolved from this script's location.
rem ---------------------------------------------------------------------------
pushd "%~dp0"
set "ROOT=%CD%"
set "PORT=7870"
set "URL=http://127.0.0.1:%PORT%"
set "PYW=%ROOT%\venv\Scripts\pythonw.exe"
set "LOG=%ROOT%\app\logs\server.log"

rem ---- keep temp files inside the project (no stray writes elsewhere) -------
if not exist "%ROOT%\tmp" mkdir "%ROOT%\tmp"
set "TMPDIR=%ROOT%\tmp"
set "TEMP=%ROOT%\tmp"
set "TMP=%ROOT%\tmp"

echo.
echo   WorldGen Studio
echo   ---------------------------------------------
echo   本地 3D 场景生成工作台
echo   项目目录: %ROOT%
echo.

rem ---- prerequisites --------------------------------------------------------
if not exist "%PYW%" (
    echo   [错误] 没有找到虚拟环境: %ROOT%\venv
    echo   请先双击运行 install.bat 完成安装。
    echo.
    pause
    popd
    exit /b 1
)
if not exist "%ROOT%\WorldGen\demo.py" (
    echo   [错误] 没有找到 WorldGen 源码: %ROOT%\WorldGen
    echo   请先双击运行 install.bat 完成安装。
    echo.
    pause
    popd
    exit /b 1
)

rem ---- already running? -----------------------------------------------------
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo   [1/2] 服务已在运行
    goto open
)

rem ---- start server ---------------------------------------------------------
echo   [1/2] 正在启动本地服务...
start "" /b "%PYW%" "%ROOT%\app\server.py"

set /a TRY=0
:wait
set /a TRY+=1
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 goto ready
if %TRY% GEQ 30 goto failed
ping -n 2 127.0.0.1 >nul
goto wait

:ready
echo         服务已就绪  %URL%
goto open

:failed
echo.
echo   [错误] 服务启动超时。
echo   日志: %LOG%
echo.
if exist "%LOG%" powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 25"
echo.
pause
popd
exit /b 1

:open
echo   [2/2] 正在打开应用窗口...
rem Prefer an Edge/Chrome --app window (frameless, looks like a native app);
rem fall back to the default browser if neither is installed.
set "BROWSER="
for %%P in (
    "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
    "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
    "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
    "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
    "%LocalAppData%\Google\Chrome\Application\chrome.exe"
) do (
    if not defined BROWSER if exist %%P set "BROWSER=%%~P"
)

if defined BROWSER (
    start "" "%BROWSER%" --app=%URL% --window-size=1520,940 --no-first-run --no-default-browser-check --user-data-dir="%ROOT%\app\.browser"
) else (
    start "" "%URL%"
)

popd
exit /b 0
