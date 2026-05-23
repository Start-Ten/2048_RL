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

echo [0/5] Detecting GPU...
:: Check for NVIDIA GPU (compatible with old and new nvidia-smi)
nvidia-smi -L >nul 2>&1
if %errorlevel% equ 0 (
    echo   NVIDIA GPU detected
    for /f "usebackq skip=1 tokens=*" %%g in (`nvidia-smi --query-gpu^=name --format^=csv 2^>nul`) do echo   GPU: %%g
    for /f "usebackq skip=1 tokens=*" %%m in (`nvidia-smi --query-gpu^=memory.total --format^=csv 2^>nul`) do echo   VRAM: %%m
) else (
    echo   No NVIDIA GPU found - CPU training only
)

echo.
echo [1/5] Checking Python + PyTorch CUDA...
python --version >nul 2>&1 || (echo ERROR: Python not found & exit /b 1)
python --version

:: Check if PyTorch has CUDA; if GPU exists but PyTorch is CPU-only, fix it
nvidia-smi -L >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import torch; assert torch.cuda.is_available(), 'CPU-only'" >nul 2>&1
    if %errorlevel% neq 0 (
        echo   NVIDIA GPU detected but PyTorch is CPU-only
        echo   Installing PyTorch with CUDA support...
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps -q
        if %errorlevel% equ 0 (
            echo   CUDA PyTorch installed. Restarting...
            :: Re-run this script
            call "%~f0" %*
            exit /b 0
        ) else (
            echo   WARNING: CUDA install failed, continuing with CPU
        )
    ) else (
        echo   PyTorch CUDA: OK
    )
) else (
    echo   PyTorch CPU mode
)

echo.
echo [2/5] Installing dependencies...
python -m pip install --quiet numpy torch tqdm matplotlib pybind11 rich pynvml
if %errorlevel% neq 0 (echo   WARNING: pip install had issues)
echo   Done.

echo.
echo [3/5] Compiling C++ engine...
python setup.py build_ext --inplace
if %errorlevel% equ 0 (
    echo   C++ engine compiled successfully
) else (
    echo   WARNING: C++ compilation failed, using Python backend
)

echo.
echo [4/5] Running environment doctor...
python doctor.py --fix
echo   Done.

echo.
echo [5/5] Starting training with TUI...
echo   Log: %LOG_FILE%
echo   Press Ctrl+C to stop.
echo.

python -u trainV4.py > "%LOG_FILE%" 2>&1

echo.
echo Training completed or interrupted.
echo Results saved in %MODEL_DIR%\
pause
