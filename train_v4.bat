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

:: Check GPU compatibility with PyTorch
nvidia-smi -L >nul 2>&1
if %errorlevel% equ 0 (
    :: Detect Blackwell GPU (RTX 50 series, sm_120)
    for /f "usebackq tokens=*" %%g in (`python -c "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.major}{p.minor}')" 2^>nul`) do set "CC=%%g"

    python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
    if %errorlevel% neq 0 (
        echo   NVIDIA GPU detected but PyTorch is CPU-only
        echo   Installing PyTorch with CUDA 12.8...
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps -q
        if %errorlevel% equ 0 (
            echo   Installed. Restarting script...
            call "%~f0" %*
            exit /b 0
        ) else (
            echo   WARNING: install failed
        )
    ) else (
        :: Check Blackwell (CC >= 120)
        if "%CC%" geq "120" (
            for /f "usebackq tokens=*" %%v in (`python -c "import torch; print(torch.__version__.split('+')[0])" 2^>nul`) do set "PT_VER=%%v"
            echo   RTX 50 series (Blackwell sm_%CC%) detected - needs PyTorch 2.7+ CUDA 12.8
            echo   Current: PyTorch !PT_VER!
            echo   Installing PyTorch CUDA 12.8...
            python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps -q
            if %errorlevel% equ 0 (
                echo   Installed. Restarting script...
                call "%~f0" %*
                exit /b 0
            ) else (
                echo   WARNING: install failed, trying nightly...
                python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps -q
                call "%~f0" %*
                exit /b 0
            )
        ) else (
            echo   PyTorch CUDA: OK (sm_%CC%)
        )
    )
) else (
    echo   No NVIDIA GPU - CPU training
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

:: Force UTF-8 encoding for TUI rendering
set PYTHONIOENCODING=utf-8
set CUDA_LAUNCH_BLOCKING=0
python -u trainV4.py > "%LOG_FILE%" 2>&1

echo.
echo Training completed or interrupted.
echo Results saved in %MODEL_DIR%\
pause
