# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DQN reinforcement learning agent for the 2048 game. Uses Dueling DQN with Prioritized Experience Replay and Double DQN to learn 2048 gameplay. Includes training scripts, a Gradio web interface for interactive play, and inference/evaluation tools.

## Commands

```bash
# Install dependencies
pip install numpy torch tqdm matplotlib gradio

# One-click training (Linux server)
bash train_v4.sh              # start fresh training
bash train_v4.sh 50000        # custom episode count
bash train_v4.sh 100000 --no-resume  # disable resume

# One-click training (Windows — blocked, wait for completion)
train_v4.bat

# Train directly
python trainV4.py             # V4 (recommended, SE-ResNet + NoisyNet + N-step)
python trainV3.py             # V3 (ResNet + mixed precision)
python trainV2.py             # V2
python trainV1.py             # V1

# Run inference — 100 games with stats + charts output to inference_results/
python model_Test.py

# Launch Gradio web app for interactive play
python 2048webapp.py

# Pull LFS files (first time after cloning)
git lfs pull
```

### Training mode control

Training scripts control behavior via the `config` (V4) or `args` (V1-V3) dict at the bottom of each file:
- `"train": 1` — train
- `"resume": 1` — resume from checkpoint
- `"play": 1` — run a single game with the trained model
- `"episodes": 200000` — training episode count

## Architecture

### Files and their roles

| File | Purpose |
|---|---|
| `game2048.py` | Standalone 4×4 game engine. Core game logic. Used by the web app. |
| `trainV1.py` / `trainV2.py` / `trainV3.py` | Self-contained training scripts (incremental improvements across versions). |
| `trainV4.py` | **Current version.** SE-ResNet + NoisyNet + N-step returns + EMA target network + CosineWarmup. ~50M parameters. |
| `model_Test.py` | Inference runner. Loads best checkpoint, runs 100 games, outputs stats/charts to `inference_results/`. |
| `2048webapp.py` | Gradio web interface. Depends on `game2048.py`. Manual play + AI move + continuous AI mode. |
| `train_v4.sh` | One-click Linux training launcher. Auto-detects GPU VRAM, configures batch size, launches with tmux/nohup. |
| `train_v4.bat` | Windows training launcher. |
| `models_v4/` | V4 model checkpoints (separate from `models/` used by V1-V3). |

### V4 Architecture (trainV4.py)

**Network (DQN_V4):** SE-ResNet + Dueling + NoisyNet
- Stem: Conv(8→256) → BN → ReLU
- Stage1: 3× SEBlock(256→256)
- Stage2: 4× SEBlock(256→512)
- Stage3: 6× SEBlock(512→1024)
- Stage4: 3× SEBlock(1024→1024)
- Value head: Conv(1024→32, 1×1) → FC(512→512) → NoisyLinear(512→1)
- Advantage head: Conv(1024→128, 1×1) → FC(2048→512) → NoisyLinear(512→4)
- Output: Q = V + A − mean(A)

**Key techniques:**
- **SEBlock:** Squeeze-and-Excitation channel attention with residual connection and BatchNorm
- **FactorizedNoisyLinear:** Gaussian noise injection for exploration (replaces epsilon-greedy)
- **N-step returns (N=3):** Accumulated over 3 steps before pushing to replay buffer
- **EMA target network:** Soft update τ=0.005 every optimizer step (no hard updates)
- **Mixed precision:** `torch.amp.autocast` + `GradScaler`
- **Cosine annealing** with linear warmup (2000 episodes) for learning rate
- **Gradient accumulation** (2 steps) for effectively larger batch sizes
- **torch.compile** auto-enabled on PyTorch 2.0+ with CUDA

**State representation (8 channels, 4×4):**
1. log2(tile_value) / 16.0
2. Empty cell mask
3. Mergeable neighbor pair mask (vectorized)
4. Max-value tile position
5. Second-max-value tile position
6. Row monotonicity indicator
7. Column monotonicity indicator
8. Max-value at corner indicator

**Training config (default):**
- episodes=200000, lr=3e-4, gamma=0.995, tau=0.005
- batch_size=1024, grad_accum_steps=2 (effective batch ~2048)
- Replay buffer: 200k entries, N-step, prioritized (α=0.6)
- Reward: score delta + empty-cell delta + sqrt(max_tile) + corner bonus + monotonicity bonus + game-over penalty

### V1-V3 Architecture (historical)

Dueling DQN with 3 convolutional layers (128 channels, 3×3 kernel, padding=1):
- Value stream: Conv(128→4, 1×1) → FC(64→128) → FC(128→1)
- Advantage stream: Conv(128→16, 1×1) → FC(256→128) → FC(128→4)
- V3 adds ResNet blocks (256→512 channels), BatchNorm, AdamW, ReduceLROnPlateau, mixed precision

## Git LFS

All files (`.py`, `.pth`, `.png`, `.txt`) are tracked by Git LFS. After cloning, run `git lfs pull` to download actual file contents. Without it, files will show as LFS pointer stubs.
