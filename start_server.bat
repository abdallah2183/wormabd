@echo off
setlocal enabledelayedexpansion

:: Set console title
title WormGPT AI Server - Starting...

echo ===========================================
echo        WormGPT AI Server - Starting
echo ===========================================
echo.

:: Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Miniconda/Anaconda is not installed or not in PATH
    echo Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Activate conda environment
echo [1/3] Activating Python environment...
call conda activate wormgpt-gpu
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to activate 'wormgpt-gpu' environment
    echo Please run 'run_me_first.bat' as Administrator first
    pause
    exit /b 1
)

:: Check Python version
echo [2/3] Checking Python and dependencies...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not available in the current environment
    pause
    exit /b 1
)

:: Check GPU support
echo [3/3] Checking GPU support...
python check_gpu.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: GPU check failed. Running in CPU mode (slow)
    set USE_GPU=0
    timeout /t 3 >nul
) else (
    set USE_GPU=1
)

:: Set environment variable for the app
if "!USE_GPU!"=="1" (
    set USE_CUDA=True
) else (
    set USE_CUDA=False
)

:: Start the server
cls
echo ===========================================
echo        WormGPT AI Server - Running
echo ===========================================
echo.
echo Server URL: http://localhost:5000
echo API Key: My_Website_Secure_Key_123456
echo GPU Acceleration: !USE_CUDA!
echo.
echo Press Ctrl+C to stop the server
echo.

:: Run the Flask app with auto-reload
set FLASK_APP=app.py
set FLASK_ENV=development
python -m flask run --host=0.0.0.0 --port=5000

:: If we get here, the server stopped
echo.
if %ERRORLEVEL% EQU 0 (
    echo Server stopped successfully
) else (
    echo Server stopped with error code: %ERRORLEVEL%
)
echo.
pause
