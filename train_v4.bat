@echo off
setlocal
cd /d "%~dp0"

set "EPISODES=%1"
if "%EPISODES%"=="" set "EPISODES=200000"

set "MODEL_DIR=models_v4"
set "LOG_DIR=logs"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%I"
set "LOG_FILE=%LOG_DIR%\train_v4_%TIMESTAMP%.log"

echo ============================================
echo   2048 DQN V4 Training (Windows)
echo ============================================
echo   Episodes: %EPISODES%
echo   Log: %LOG_FILE%
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1 || (echo ERROR: Python not found & exit /b 1)
python --version

echo.
echo [2/4] Installing dependencies...
python -m pip install --quiet numpy torch tqdm matplotlib pybind11
if %errorlevel% neq 0 (echo   WARNING: pip install had issues, continuing...)
echo   Done.

echo.
echo [3/4] Compiling C++ engine...
python setup.py build_ext --inplace
if %errorlevel% equ 0 (
    echo   C++ engine compiled successfully
) else (
    echo   WARNING: C++ compilation failed, using Python backend
)

echo.
echo [4/4] Starting training...
echo   Logging to: %LOG_FILE%
echo   Press Ctrl+C to stop.
echo.

python -u trainV4.py > "%LOG_FILE%" 2>&1

echo.
echo Training completed or interrupted.
echo Results saved in %MODEL_DIR%\
pause
