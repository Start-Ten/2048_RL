# CLAUDE.md

DQN RL agent for 2048 game. SE-ResNet (~190M params) + NoisyNet + N-step returns + EMA target.

## Commands

```bash
pip install -r requirements.txt

# Train
python main.py train                          # default 200k episodes, batch mode
python main.py train --episodes 50000 --no-cpp  # single-env, no C++
python main.py train --resume --n-envs 128    # resume with 128 parallel envs

# Play / Web / Eval / Doctor
python main.py play                           # interactive game with model
python main.py web                            # Gradio web UI
python main.py test --games 100               # evaluation
python main.py doctor --fix                   # auto-fix environment
```

## Project Structure

```
main.py              # CLI entry (train/play/web/test/doctor)
setup.py             # C++ extension build
game2048_cpp.cpp     # C++ engine + BatchGame2048 (pybind11)
requirements.txt

src/
  engine.py          # Game2048 (NumPy, 8-channel state)
  networks.py        # DQN_V4, SEBlock, FactorizedNoisyLinear
  replay.py          # NStepReplayBuffer (per-env, prioritized)
  agent.py           # DQNAgent (train step, EMA target, mixed precision)
  scheduler.py       # CosineWarmupScheduler
  trainer.py         # train() — auto-selects batch or single-env

ui/
  tui.py             # Rich TUI (GPU util/VRAM/power/temp, sparklines)

tools/
  doctor.py          # Environment diagnostics & auto-fix
  inference.py       # Standalone evaluation

web/
  app.py             # Gradio web interface
```

## Architecture

**DQN_V4:** SE-ResNet backbone (256→512→1024, 16 SE blocks) + Dueling + NoisyNet.
State: 8 channels (log2 tile, empty, mergeable, max pos, 2nd max, row/col monotonic, corner).

**Training:** N-step (N=3) prioritized replay (200k, α=0.6), EMA target (τ=0.005),
CosineWarmup LR, gradient accumulation, optional C++ batch engine (N parallel envs).
