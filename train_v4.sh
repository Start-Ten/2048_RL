#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 2048 DQN V4 一键训练脚本（含C++编译 + 批量并行环境）
# 用法: bash train_v4.sh [episodes] [--no-cpp] [--resume]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
MODEL_DIR="models_v4"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_v4_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_v4.pid"

# ---- 参数解析 ----
EPISODES="${1:-200000}"
NO_CPP=""
RESUME_FLAG=""
NO_RESUME=""
for arg in "$@"; do
    if [ "$arg" = "--no-cpp" ]; then NO_CPP="1"; fi
    if [ "$arg" = "--resume" ]; then RESUME_FLAG="1"; fi
    if [ "$arg" = "--no-resume" ]; then NO_RESUME="1"; fi
done

echo "============================================"
echo "  2048 DQN V4 Training Launcher"
echo "============================================"
echo "  Episodes:      ${EPISODES}"
echo "  C++ engine:    $([ "$NO_CPP" = "1" ] && echo 'DISABLED' || echo 'auto')"
echo "  Timestamp:     ${TIMESTAMP}"
echo "  Log file:      ${LOG_FILE}"
echo "============================================"

# ---- 环境检测 ----
echo ""
echo "[0/5] 检测环境..."

PYTHON=$(command -v python3 || command -v python || echo "")
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install Python 3.10+ first."
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
echo "  Python: ${PY_VER} ($PYTHON)"

CUDA_AVAIL=$($PYTHON -c "
import torch
print(f'PyTorch {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')
    total_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'VRAM: {total_gb:.1f} GB')
else:
    print('WARNING: CUDA not available — training will be slow on CPU')
" 2>&1 || true)
echo "$CUDA_AVAIL" | while IFS= read -r line; do echo "  $line"; done

# ---- Doctor: 环境诊断与自动修复 ----
echo ""
echo "[*] 运行环境诊断 (doctor)..."
$PYTHON doctor.py --fix 2>&1 || echo "  doctor: some issues remain, continuing..."

# ---- 安装依赖 ----
echo ""
echo "[1/5] 安装 Python 依赖..."
$PYTHON -m pip install --quiet numpy torch tqdm matplotlib pybind11 2>&1 | tail -1 || {
    echo "  pip install had issues, continuing..."
}
echo "  依赖安装完成"

# ---- 编译 C++ 加速引擎 ----
if [ "$NO_CPP" != "1" ]; then
    echo ""
    echo "[2/5] 编译 C++ 加速引擎..."
    if $PYTHON setup.py build_ext --inplace 2>&1 | tail -3; then
        echo "  C++ engine compiled successfully"
    else
        echo "  WARNING: C++ compilation failed, falling back to Python backend"
        NO_CPP="1"
    fi
else
    echo ""
    echo "[2/5] 跳过 C++ 编译 (--no-cpp)"
fi

# ---- 创建目录 ----
echo ""
echo "[3/5] 创建目录..."
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"
echo "  models_v4/ ✓"
echo "  logs/ ✓"

# ---- 自动配置 GPU 参数 ----
HAS_GPU=$($PYTHON -c "import torch; print('1' if torch.cuda.is_available() else '0')")

# Triton (torch.compile 加速，仅 Linux CUDA)
if [ "$HAS_GPU" = "1" ] && [ "$(uname -s)" = "Linux" ]; then
    echo ""
    echo "[*] 安装 Triton (torch.compile 加速)..."
    $PYTHON -m pip install --quiet triton 2>/dev/null && echo "  triton: installed" || echo "  triton: skipped"
fi

if [ "$HAS_GPU" = "1" ]; then
    VRAM_GB=$($PYTHON -c "
import torch
total = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'{total:.0f}')
")
    if [ "$VRAM_GB" -ge 40 ]; then
        BATCH_SIZE=2048
        GRAD_ACCUM=2
        N_ENVS=128
    elif [ "$VRAM_GB" -ge 20 ]; then
        BATCH_SIZE=1024
        GRAD_ACCUM=2
        N_ENVS=64
    elif [ "$VRAM_GB" -ge 10 ]; then
        BATCH_SIZE=512
        GRAD_ACCUM=2
        N_ENVS=32
    else
        BATCH_SIZE=256
        GRAD_ACCUM=2
        N_ENVS=16
    fi
    echo ""
    echo "  GPU VRAM: ~${VRAM_GB}GB → batch_size=${BATCH_SIZE}, n_envs=${N_ENVS}"
else
    BATCH_SIZE=128
    GRAD_ACCUM=1
    N_ENVS=1
    NO_CPP="1"
    echo ""
    echo "  CPU模式 → batch_size=${BATCH_SIZE}"
fi

# ---- 应用配置到 trainV4.py ----
echo ""
echo "[4/5] 应用配置..."

$PYTHON -c "
import re

with open('trainV4.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = re.sub(r'\"episodes\":\s*\d+', f'\"episodes\": ${EPISODES}', code)
code = re.sub(r'\"batch_size\":\s*\d+', f'\"batch_size\": ${BATCH_SIZE}', code)
code = re.sub(r'\"grad_accum_steps\":\s*\d+', f'\"grad_accum_steps\": ${GRAD_ACCUM}', code)
code = re.sub(r'\"n_envs\":\s*\d+', f'\"n_envs\": ${N_ENVS}', code)

if '${NO_CPP}' == '1':
    code = re.sub(r'\"use_batch\":\s*\d+', '\"use_batch\": 0', code)
else:
    code = re.sub(r'\"use_batch\":\s*\d+', '\"use_batch\": 1', code)

if '${RESUME_FLAG}' == '1':
    code = re.sub(r'\"resume\":\s*\d+', '\"resume\": 1', code)
elif '${NO_RESUME}' == '1':
    code = re.sub(r'\"resume\":\s*\d+', '\"resume\": 0', code)

with open('trainV4.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('  配置已写入 trainV4.py')
"

# ---- 启动训练 ----
echo ""
echo "[5/5] 启动训练..."
echo "  日志: ${LOG_FILE}"
echo ""

if command -v tmux &> /dev/null; then
    echo "  使用 tmux 会话 'train2048' 启动训练"
    echo "  查看进度: tmux attach -t train2048"
    echo "  退出查看: Ctrl+B, 然后按 D"
    echo ""
    tmux new-session -d -s train2048 "$PYTHON -u trainV4.py 2>&1 | tee ${LOG_FILE}"
    echo "  ✓ 训练已在 tmux 后台启动"
else
    nohup $PYTHON -u trainV4.py > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$PID_FILE"
    echo "  ✓ 训练已在后台启动"
    echo "  PID: ${TRAIN_PID}"
    echo "  查看日志: tail -f ${LOG_FILE}"
    echo "  终止训练: kill ${TRAIN_PID}"
fi

echo ""
echo "============================================"
echo "  训练配置:"
echo "    Episodes:   ${EPISODES}"
echo "    Batch:      ${BATCH_SIZE} × ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
if [ "$NO_CPP" != "1" ]; then
    echo "    并行环境:   ${N_ENVS}"
    echo "    引擎:       C++ (批量)"
else
    echo "    引擎:       Python (单环境)"
fi
echo ""
echo "  训练进度监控:"
echo "    tail -f ${LOG_FILE}"
echo ""
echo "  模型保存位置: ${MODEL_DIR}/"
echo "     dqn_2048.pth           - 定期保存"
echo "     dqn_2048_best_tile.pth - 最佳方块"
echo "     dqn_2048_best_score.pth- 最佳分数"
echo "     checkpoint.pth         - 断点续训"
echo "============================================"
