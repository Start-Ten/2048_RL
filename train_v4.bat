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
echo   2048 DQN V4 Training ^(Windows^)
echo ============================================
echo   Episodes: %EPISODES%
echo   Log: %LOG_FILE%
echo.

echo [0/5] Detecting GPU...
nvidia-smi -L >nul 2>&1
if %errorlevel% equ 0 (
    echo   NVIDIA GPU detected
    for /f "usebackq skip=1 tokens=*" %%g in (`nvidia-smi --query-gpu^=name --format^=csv 2^>nul`) do echo   GPU: %%g
) else (
    echo   No NVIDIA GPU found
)

echo.
echo [1/5] Checking Python + PyTorch CUDA...
python --version >nul 2>&1 || (echo ERROR: Python not found & exit /b 1)
python --version

:: Use a Python helper to do all GPU/PyTorch compatibility checks at once
python -c "
import subprocess, sys

def check_gpu():
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    return result.returncode == 0

def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            ver = torch.__version__
            cc = f'{props.major}{props.minor}'
            return {'ok': True, 'cc': cc, 'pt_ver': ver, 'cuda_ver': torch.version.cuda or '0'}
        return {'ok': False, 'reason': 'CPU-only PyTorch'}
    except ImportError:
        return {'ok': False, 'reason': 'PyTorch not installed'}

has_gpu = check_gpu()
if not has_gpu:
    print('NO_GPU')
    sys.exit(0)

info = check_cuda()
if not info['ok']:
    print(f'NEED_CUDA:{info[\"reason\"]}')
elif int(info['cc']) >= 120:
    # Check if current PyTorch supports Blackwell (needs 2.7+ with CUDA 12.8+)
    pt_ver = tuple(int(x) for x in info['pt_ver'].split('.')[:2])
    cu_ver = float(info['cuda_ver'])
    if pt_ver < (2, 7) or cu_ver < 12.8:
        print(f'BLACKWELL:{info[\"cc\"]}:{info[\"pt_ver\"]}')
    else:
        print(f'OK:{info[\"cc\"]}')
else:
    print(f'OK:{info[\"cc\"]}')
" > "%TEMP%\gpu_check.txt" 2>nul

set /p GPU_STATUS=<"%TEMP%\gpu_check.txt"
del "%TEMP%\gpu_check.txt" 2>nul

if "%GPU_STATUS%"=="NO_GPU" (
    echo   No NVIDIA GPU - CPU training
) else if "%GPU_STATUS%"=="NEED_CUDA:CPU-only PyTorch" (
    echo   GPU detected but PyTorch is CPU-only
    echo   Installing PyTorch CUDA 12.8...
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps -q
    echo   Restarting...
    call "%~f0" %*
    exit /b 0
) else if "%GPU_STATUS:~0,9%"=="BLACKWELL" (
    for /f "tokens=2,3 delims=:" %%a in ("%GPU_STATUS%") do (
        echo   RTX 50 series - Blackwell sm_%%a - needs PyTorch 2.7+ CUDA 12.8
        echo   Current PyTorch: %%b
    )
    echo   Installing PyTorch CUDA 12.8...
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps -q 2>nul
    echo   Restarting...
    call "%~f0" %*
    exit /b 0
) else (
    echo   PyTorch CUDA: OK
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

set PYTHONIOENCODING=utf-8
python -u trainV4.py > "%LOG_FILE%" 2>&1

echo.
echo Training completed or interrupted.
echo Results saved in %MODEL_DIR%\
pause
