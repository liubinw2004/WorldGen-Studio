@echo off
rem ===========================================================================
rem  WorldGen Studio - launcher
rem  Starts the local server and opens the app window.
rem  Paths are resolved from this script's own location, so the project can
rem  live anywhere.
rem
rem  NOTE: ASCII-only on purpose -- cmd.exe mis-parses UTF-8 Chinese in .bat.
rem ===========================================================================
setlocal EnableExtensions
title WorldGen Studio

pushd "%~dp0"
set "ROOT=%CD%"
set "PORT=7870"
set "URL=http://127.0.0.1:%PORT%"
set "PYW=%ROOT%\venv\Scripts\pythonw.exe"
set "LOG=%ROOT%\app\logs\server.log"

rem ---- keep temp files inside the project ----------------------------------
if not exist "%ROOT%\tmp" mkdir "%ROOT%\tmp"
set "TMPDIR=%ROOT%\tmp"
set "TEMP=%ROOT%\tmp"
set "TMP=%ROOT%\tmp"

echo.
echo   WorldGen Studio
echo   ---------------------------------------------
echo   Project root: %ROOT%
echo.

rem ---- prerequisites --------------------------------------------------------
if not exist "%PYW%" (
    echo   [ERROR] Virtual environment not found: %ROOT%\venv
    echo   Run install.bat first.
    echo.
    pause
    popd
    exit /b 1
)
if not exist "%ROOT%\WorldGen\demo.py" (
    echo   [ERROR] WorldGen sources not found: %ROOT%\WorldGen
    echo   Run install.bat first.
    echo.
    pause
    popd
    exit /b 1
)

rem ---- already running? -----------------------------------------------------
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo   [1/2] Server already running
    goto open
)

rem ---- start server ---------------------------------------------------------
echo   [1/2] Starting local server ...
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
echo         Server ready at %URL%
goto open

:failed
echo.
echo   [ERROR] The server did not start in time.
echo   Log: %LOG%
echo.
if exist "%LOG%" powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 25"
echo.
pause
popd
exit /b 1

:open
echo   [2/2] Opening the app window ...
rem Prefer an Edge/Chrome --app window (frameless, feels native);
rem fall back to the default browser.
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
