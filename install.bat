@echo off
chcp 65001 >nul
setlocal EnableExtensions
title WorldGen Studio - 安装

rem ===========================================================================
rem  WorldGen Studio 一键安装
rem
rem  用法:
rem    install.bat            正常安装（官方源）
rem    install.bat --mirror   国内网络（hf-mirror + GitHub 代理 + 清华/腾讯 PyPI）
rem
rem  这个脚本只做三件事：找到 Python、建虚拟环境、把剩下交给
rem  scripts\bootstrap.py。模型权重不在安装范围内，装完会打印清单。
rem ===========================================================================

pushd "%~dp0"
set "ROOT=%CD%"

echo.
echo   WorldGen Studio 安装程序
echo   ---------------------------------------------
echo   项目目录: %ROOT%
echo.

rem ---- 参数 ----------------------------------------------------------------
set "MIRROR_ARG="
if /I "%~1"=="--mirror" set "MIRROR_ARG=--mirror"
if defined MIRROR_ARG (
    echo   模式: 国内镜像
) else (
    echo   模式: 官方源 ^(国内网络请改用 install.bat --mirror^)
)
echo.

rem ---- 1. 定位 Python 3.11/3.12 -------------------------------------------
echo   [1/4] 查找 Python 3.11 / 3.12 ...
set "BASE_PY="

py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "BASE_PY=py -3.12"

if not defined BASE_PY (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "BASE_PY=py -3.11"
)

if not defined BASE_PY (
    for /f "tokens=2" %%v in ('python -V 2^>^&1') do (
        echo %%v | findstr /B /C:"3.12" >nul && set "BASE_PY=python"
        echo %%v | findstr /B /C:"3.11" >nul && set "BASE_PY=python"
    )
)

if not defined BASE_PY (
    echo.
    echo   [错误] 没有找到 Python 3.11 或 3.12。
    echo.
    echo   请先安装 Python 3.12:  https://www.python.org/downloads/
    echo   安装时务必勾选 "Add python.exe to PATH"。
    echo   注意: 3.13 无法使用本项目（torch cu128 / nunchaku 只提供 cp312 轮子）。
    echo.
    pause
    popd
    exit /b 1
)

for /f "delims=" %%v in ('%BASE_PY% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%v"
echo         使用 Python %PYVER%  ^(%BASE_PY%^)

rem ---- 2. 生成 env 配置 ----------------------------------------------------
echo   [2/4] 生成环境配置 ...
%BASE_PY% "%ROOT%\config.py" --emit >nul
if errorlevel 1 (
    echo   [错误] 生成配置失败。
    pause
    popd
    exit /b 1
)
call "%ROOT%\env\set_env.bat"
set "PIP_CONFIG_FILE=%ROOT%\env\pip.ini"
if defined MIRROR_ARG set "WORLDGEN_MIRROR=1"

rem ---- 3. 创建虚拟环境 -----------------------------------------------------
echo   [3/4] 准备虚拟环境 ...
if exist "%ROOT%\venv\Scripts\python.exe" (
    echo         虚拟环境已存在，跳过创建
) else (
    %BASE_PY% -m venv "%ROOT%\venv"
    if errorlevel 1 (
        echo   [错误] 创建虚拟环境失败。
        pause
        popd
        exit /b 1
    )
    echo         已创建 venv
)

"%ROOT%\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo   [警告] pip 升级失败，继续尝试安装依赖。
)

rem ---- 4. 安装依赖 ---------------------------------------------------------
echo   [4/4] 安装依赖（首次约 10-20 分钟，取决于网速）...
echo.
"%ROOT%\venv\Scripts\python.exe" "%ROOT%\scripts\bootstrap.py" %MIRROR_ARG%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   安装流程结束。请按上面清单放好模型，然后双击 start.bat。
) else (
    echo   安装未完成，请查看上面的错误信息。
    echo   修好后可以重复运行本脚本，已完成的步骤会自动跳过。
)
echo.
pause
popd
exit /b %RC%
