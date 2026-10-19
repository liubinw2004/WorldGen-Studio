@echo off
REM ===========================================================================
REM run_bootstrap.bat - idempotent end-to-end WorldGen reproduction (Windows).
REM
REM This is a thin wrapper: the real orchestrator is run_bootstrap.sh (Git Bash),
REM which is state.json-driven and skips completed steps. Keeping ONE orchestrator
REM avoids bash/cmd drift.
REM
REM Usage:
REM   run_bootstrap.bat                 REM T01..T03b + T05 (no demo)
REM   run_bootstrap.bat --with-demo     REM also run T04 (needs GPU)
REM   run_bootstrap.bat --list          REM show step status only
REM ===========================================================================
setlocal
where bash >nul 2>nul
if errorlevel 1 (
    echo [run_bootstrap] FATAL: Git Bash (bash.exe) not found on PATH.
    exit /b 3
)
bash /d/mm/scripts/run_bootstrap.sh %*
set "RC=%errorlevel%"
endlocal & exit /b %RC%
