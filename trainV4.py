"""
trainV4.py — 2048 DQN 强化学习训练脚本 (大模型升级版)

相对于V3的改进：
- SE-ResNet 骨干网络 (256→512→1024 通道，含Squeeze-and-Excitation注意力)
- Noisy Networks 替代 epsilon-greedy 探索 (Factorized Gaussian Noise)
- Multi-step returns (N=3) 更准确的价值估计
- EMA 软更新目标网络 (τ=0.005)
- torch.compile JIT 编译加速 (PyTorch 2.0+)
- Cosine Annealing + Linear Warmup 学习率调度
- 梯度累积 (2步) 等效更大batch
- 8通道状态表示 (新增二阶最大值、单调性、角落奖励等特征)
- 回放缓冲区容量 200k
- 改进的奖励函数 (角落奖励 + 单调性奖励)
- 向量化游戏引擎 (大幅减少Python循环，预分配数组)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import sys
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from collections import deque
from contextlib import nullcontext

# ---- TUI 监控 (可选) ----
try:
    from training_tui import create_monitor
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False
    create_monitor = None

# ---- C++ 加速引擎 (可选) ----
try:
    # 添加 MinGW DLL 路径（Windows 下 MinGW 编译时需要）
    import os as _os
    _mingw_paths = [
        r"C:\mingw64\bin", r"C:\Windows\mingw64\bin",
        r"C:\msys64\mingw64\bin", r"C:\msys64\ucrt64\bin"
    ]
    for _p in _mingw_paths:
        if _os.path.isdir(_p) and _p not in _os.environ.get("PATH", ""):
            _os.add_dll_directory(_p)
    import game2048_cpp
    CPP_AVAILABLE = True
    print("game2048_cpp: loaded (C++ accelerated)")
except ImportError:
    game2048_cpp = None
    CPP_AVAILABLE = False
    print("game2048_cpp: not available, using Python backend")

# ============================================================
# 设备检测
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_str = "cuda"
elif torch.xpu.is_available():
    device = torch.device("xpu")
    device_str = "xpu"
else:
    device = torch.device("cpu")
    device_str = "cpu"
print(f"Using device: {device}")

# PyTorch 版本检测
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
USE_COMPILE = TORCH_VERSION >= (2, 0) and device.type == "cuda"
# 自动安装 triton（torch.compile 依赖，仅 Linux）
if USE_COMPILE:
    try:
        import triton
    except ImportError:
        import subprocess, platform
        if platform.system() == "Linux":
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "triton", "-q"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                import triton
                print("triton installed automatically")
            except Exception:
                USE_COMPILE = False
                print("triton install failed, torch.compile disabled")
        else:
            USE_COMPILE = False
            print("triton not available on {}, torch.compile disabled".format(platform.system()))

# ============================================================
# 2048 游戏环境 (向量化高速版)
# ============================================================
class Game2048:
    """2048游戏引擎 — 向量化优化，最小化 Python 循环"""

    def __init__(self, size=4):
        self.size = size
        self._state_buf = np.zeros((8, size, size), dtype=np.float32)
        self._slide_bufs = [np.zeros(size, dtype=np.int32) for _ in range(4)]
        self._score = 0
        self._board = np.zeros((size, size), dtype=np.int32)
        self.game_over = False
        self.reset()

    @property
    def score(self):
        return self._score

    @property
    def board(self):
        return self._board

    def reset(self):
        self._board.fill(0)
        self._score = 0
        self.game_over = False
        self._add_tile()
        self._add_tile()
        return self.get_state()

    def _add_tile(self):
        flat = self._board.ravel()
        zeros = np.flatnonzero(flat == 0)
        if len(zeros) > 0:
            idx = zeros[random.randint(0, len(zeros) - 1)]
            flat[idx] = 2 if random.random() < 0.9 else 4

    def _slide_row_left(self, row):
        """高速行滑动 — 结果写入 _slide_buf，返回 (任意移动, 得分增量)"""
        nz = row[row != 0]
        if len(nz) == 0:
            return False, 0

        buf = self._slide_bufs[0]  # 复用缓冲区
        idx = 0
        merged = False
        score_inc = 0
        i = 0
        n = len(nz)

        while i < n:
            if i + 1 < n and nz[i] == nz[i + 1]:
                merged_val = nz[i] * 2
                buf[idx] = merged_val
                score_inc += merged_val
                merged = True
                i += 2
            else:
                buf[idx] = nz[i]
                i += 1
            idx += 1

        buf[idx:] = 0

        # 检查是否移动：非零元素左对齐时，若前方有0则发生了移动
        original_positions = np.flatnonzero(row)
        if len(original_positions) > 0:
            moved = (original_positions[0] != 0 or merged)
        else:
            moved = False

        return moved or merged, score_inc

    def move(self, direction):
        original_board = self._board.copy()
        old_score = self._score
        moved = False

        if direction == 0:  # 上
            for j in range(4):
                mv, sc = self._slide_row_left(self._board[:, j])
                if mv:
                    moved = True
                    self._board[:, j] = self._slide_bufs[0]
                self._score += sc
        elif direction == 1:  # 右 (反转→左滑→反转)
            for i in range(4):
                mv, sc = self._slide_row_left(self._board[i, ::-1].copy())
                if mv:
                    moved = True
                    self._board[i, :] = self._slide_bufs[0][::-1]
                self._score += sc
        elif direction == 2:  # 下
            for j in range(4):
                mv, sc = self._slide_row_left(self._board[::-1, j].copy())
                if mv:
                    moved = True
                    self._board[:, j] = self._slide_bufs[0][::-1]
                self._score += sc
        elif direction == 3:  # 左
            for i in range(4):
                mv, sc = self._slide_row_left(self._board[i, :])
                if mv:
                    moved = True
                    self._board[i, :] = self._slide_bufs[0]
                self._score += sc

        if moved:
            self._add_tile()
            self.game_over = self._is_game_over()

        reward = self._calc_reward(old_score, original_board)
        return self.get_state(), reward, self.game_over

    def _is_game_over(self):
        b = self._board
        if np.any(b == 0):
            return False
        # 向量化检查
        if np.any(b[:, :-1] == b[:, 1:]):
            return False
        if np.any(b[:-1, :] == b[1:, :]):
            return False
        return True

    def _calc_reward(self, old_score, original_board):
        score_reward = (self._score - old_score) * 0.1

        empty_before = np.count_nonzero(original_board == 0)
        empty_after = np.count_nonzero(self._board == 0)
        empty_reward = (empty_after - empty_before) * 0.1
        if empty_after < 5 and empty_after != empty_before:
            empty_reward -= np.square(5 - empty_after) * 0.15

        max_before = original_board.max()
        max_after = self._board.max()
        max_tile_reward = 0.0
        if max_after > max_before:
            max_tile_reward = math.sqrt(max_after) * 1.5

        # 角落奖励
        corner_reward = 0.0
        if max_after >= 256:
            if self._board[0, 0] == max_after or self._board[0, -1] == max_after or \
               self._board[-1, 0] == max_after or self._board[-1, -1] == max_after:
                corner_reward = 0.5

        # 单调性奖励 (向量化)
        monotonicity_reward = 0.0
        for i in range(4):
            row_nz = self._board[i][self._board[i] > 0]
            if len(row_nz) >= 3:
                diffs = np.diff(row_nz)
                if np.all(diffs >= 0) or np.all(diffs <= 0):
                    monotonicity_reward += 0.02
            col_nz = self._board[:, i][self._board[:, i] > 0]
            if len(col_nz) >= 3:
                diffs = np.diff(col_nz)
                if np.all(diffs >= 0) or np.all(diffs <= 0):
                    monotonicity_reward += 0.02

        game_over_penalty = -15.0 if self.game_over else 0.0

        return (score_reward + empty_reward + max_tile_reward +
                corner_reward + monotonicity_reward + game_over_penalty)

    def get_state(self):
        """8通道状态表示 (向量化版本)"""
        state = self._state_buf
        b = self._board

        state.fill(0.0)

        # 通道0: log2(tile) 归一化
        pos = b > 0
        state[0][pos] = np.log2(b[pos]) * (1.0 / 16.0)

        # 通道1: 空格子
        state[1] = (b == 0).astype(np.float32)

        # 通道2: 邻居可合并指示 (向量化)
        # 水平相等
        horiz = (b[:, :-1] > 0) & (b[:, :-1] == b[:, 1:])
        state[2, :, :-1] = horiz.astype(np.float32)
        state[2, :, 1:] += horiz.astype(np.float32)
        # 垂直相等
        vert = (b[:-1, :] > 0) & (b[:-1, :] == b[1:, :])
        state[2, :-1, :] += vert.astype(np.float32)
        state[2, 1:, :] += vert.astype(np.float32)

        # 通道3: 最大值位置
        max_val = b.max()
        if max_val > 0:
            state[3] = (b == max_val).astype(np.float32)

            # 通道4: 第二大的值位置
            second_mask = (b > 0) & (b < max_val)
            if second_mask.any():
                second_val = b[second_mask].max()
                state[4] = (b == second_val).astype(np.float32)

            # 通道7: 最大值在角落
            if (b[0, 0] == max_val or b[0, -1] == max_val or
                b[-1, 0] == max_val or b[-1, -1] == max_val):
                max_pos = np.unravel_index(b.argmax(), (4, 4))
                state[7, max_pos[0], max_pos[1]] = 1.0

        # 通道5/6: 行/列单调递减
        for i in range(4):
            row_nz = b[i][b[i] > 0]
            if len(row_nz) >= 2:
                state[5, i, :] = float(np.all(np.diff(row_nz) <= 0))
            col_nz = b[:, i][b[:, i] > 0]
            if len(col_nz) >= 2:
                state[6, :, i] = float(np.all(np.diff(col_nz) <= 0))

        return state.copy()

    def get_valid_moves(self):
        """快速有效移动检查 — 单次遍历棋盘"""
        b = self._board
        can_up = can_right = can_down = can_left = False

        has_empty = b.min() == 0  # 快速检查是否有空位

        for i in range(4):
            for j in range(4):
                v = b[i, j]
                if v == 0:
                    continue
                if not can_up and i > 0 and (b[i-1, j] == 0 or b[i-1, j] == v):
                    can_up = True
                if not can_down and i < 3 and (b[i+1, j] == 0 or b[i+1, j] == v):
                    can_down = True
                if not can_left and j > 0 and (b[i, j-1] == 0 or b[i, j-1] == v):
                    can_left = True
                if not can_right and j < 3 and (b[i, j+1] == 0 or b[i, j+1] == v):
                    can_right = True
                if can_up and can_down and can_left and can_right:
                    return [0, 1, 2, 3]

        if not has_empty:
            # 所有格子已满，重新检查合并可能性（非零到零的方向无效）
            can_up = can_down = can_left = can_right = False
            for i in range(4):
                for j in range(4):
                    v = b[i, j]
                    if not can_up and i > 0 and b[i-1, j] == v:
                        can_up = True
                    if not can_down and i < 3 and b[i+1, j] == v:
                        can_down = True
                    if not can_left and j > 0 and b[i, j-1] == v:
                        can_left = True
                    if not can_right and j < 3 and b[i, j+1] == v:
                        can_right = True

        valid = []
        if can_up: valid.append(0)
        if can_right: valid.append(1)
        if can_down: valid.append(2)
        if can_left: valid.append(3)
        return valid


# ============================================================
# 神经网络模块
# ============================================================

class FactorizedNoisyLinear(nn.Module):
    """分解高斯噪声线性层 (Rainbow DQN)"""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        epsilon_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力残差块"""

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE 注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, max(1, out_channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, out_channels // reduction), out_channels),
            nn.Sigmoid()
        )

        # shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        # SE 通道注意力
        se_weight = self.se(out).view(out.size(0), -1, 1, 1)
        out = out * se_weight
        out = out + residual
        return F.relu(out, inplace=True)


class DQN_V4(nn.Module):
    """SE-ResNet + Dueling + NoisyNet 架构"""

    def __init__(self, input_channels=8, output_size=4):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Stage1: 256→256, 3 blocks
        self.stage1 = nn.Sequential(
            SEBlock(256, 256),
            SEBlock(256, 256),
            SEBlock(256, 256),
        )

        # Stage2: 256→512, 4 blocks
        self.stage2 = nn.Sequential(
            SEBlock(256, 512),
            SEBlock(512, 512),
            SEBlock(512, 512),
            SEBlock(512, 512),
        )

        # Stage3: 512→1024, 6 blocks
        self.stage3 = nn.Sequential(
            SEBlock(512, 1024),
            SEBlock(1024, 1024),
            SEBlock(1024, 1024),
            SEBlock(1024, 1024),
            SEBlock(1024, 1024),
            SEBlock(1024, 1024),
        )

        # Stage4: 1024→1024, 3 blocks
        self.stage4 = nn.Sequential(
            SEBlock(1024, 1024),
            SEBlock(1024, 1024),
            SEBlock(1024, 1024),
        )

        # Dueling 头
        # 价值流
        self.value_conv = nn.Conv2d(1024, 32, kernel_size=1)
        self.value_fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 512),
            nn.ReLU(inplace=True),
        )
        self.value_noisy = FactorizedNoisyLinear(512, 1, sigma_init=0.5)

        # 优势流
        self.advantage_conv = nn.Conv2d(1024, 128, kernel_size=1)
        self.advantage_fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
        )
        self.advantage_noisy = FactorizedNoisyLinear(512, output_size, sigma_init=0.5)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        value = self.value_conv(x)
        value = self.value_fc(value)
        value = self.value_noisy(value)

        advantage = self.advantage_conv(x)
        advantage = self.advantage_fc(advantage)
        advantage = self.advantage_noisy(advantage)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()


# ============================================================
# N-step 经验回放缓冲区
# ============================================================
class NStepPrioritizedReplayBuffer:
    def __init__(self, capacity=200000, alpha=0.6, n_step=3, gamma=0.995, n_envs=1):
        self.capacity = capacity
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
        self.size = 0
        # Per-environment n-step buffers (prevents interleaving in batch mode)
        self.n_step_buffer = [deque(maxlen=n_step) for _ in range(max(1, n_envs))]

    def push(self, state, action, reward, next_state, done, env_id=0):
        # env_id: for batch mode, each env has its own n-step buffer
        n_step_buf = self.n_step_buffer[env_id]
        n_step_buf.append((state, action, reward, next_state, done))
        if done:
            # Episode ended: flush all remaining transitions for this env
            while len(n_step_buf) >= 1:
                s0, a0, _, _, _ = n_step_buf[0]
                r_acc = 0.0
                for t in range(len(n_step_buf)):
                    r_acc += (self.gamma ** t) * n_step_buf[t][2]
                _, _, _, ns_last, d_last = n_step_buf[-1]
                self._store(s0, a0, r_acc, ns_last, d_last)
                n_step_buf.popleft()
            return
        if len(n_step_buf) < self.n_step:
            return
        # Compute n-step return
        s0, a0, _, _, _ = n_step_buf[0]
        r_acc = 0.0
        for t in range(self.n_step):
            r_acc += (self.gamma ** t) * n_step_buf[t][2]
        _, _, _, ns_n, d_n = n_step_buf[self.n_step - 1]
        self._store(s0, a0, r_acc, ns_n, d_n)
        n_step_buf.popleft()  # Slide window

    def _store(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        entry = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return None

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states)),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(np.array(next_states)),
            torch.tensor(dones, dtype=torch.float),
            indices,
            torch.tensor(weights)
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-6)

    def __len__(self):
        return self.size


# ============================================================
# DQN 智能体
# ============================================================
class DQNAgent:
    def __init__(self, input_channels=8, action_size=4,
                 lr=3e-4, gamma=0.995, tau=0.005,
                 batch_size=512, grad_accum_steps=2):
        self.input_channels = input_channels
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau  # EMA 系数
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps

        self.policy_net = DQN_V4(input_channels, action_size).to(device)
        self.target_net = DQN_V4(input_channels, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # EMA: 初始化 target net 不需要 grad
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-4
        )
        self.memory = NStepPrioritizedReplayBuffer(
            capacity=200000, alpha=0.6, n_step=3, gamma=gamma, n_envs=128
        )
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # torch.compile
        self.compiled = False
        if USE_COMPILE:
            try:
                self.policy_net = torch.compile(
                    self.policy_net, mode="default"
                )
                self.compiled = True
                print("torch.compile enabled for policy_net")
            except Exception as e:
                print(f"torch.compile failed: {e}, continuing without compile")

        # 混合精度 (仅CUDA/XPU，CPU不支持)
        self.use_amp = device.type in ("cuda", "xpu")
        self.scaler = torch.amp.GradScaler(device_str) if self.use_amp else None

        self.steps_done = 0
        self.grad_step = 0

    def select_action(self, state, valid_moves, evaluate=False):
        self.steps_done += 1

        with torch.no_grad():
            if not evaluate:
                self.policy_net.reset_noise()
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()

        valid_q = np.full(self.action_size, -np.inf)
        for move in valid_moves:
            valid_q[move] = q_values[move]

        return int(np.argmax(valid_q))

    def optimize_model(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return 0.0

        sample = self.memory.sample(self.batch_size, beta)
        if sample is None:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = sample
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        weights = weights.to(device)

        # 每一步优化时重置噪声
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # 前向：仅在GPU上使用autocast
        amp_ctx = torch.amp.autocast(device_str) if self.use_amp else nullcontext()
        with amp_ctx:
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_actions = self.policy_net(next_states).max(1)[1]
                next_q = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze()
                target_q = rewards + (1.0 - dones) * self.gamma * next_q

            losses = self.loss_fn(current_q, target_q)
            loss = (losses * weights).mean()
            loss = loss / self.grad_accum_steps

        # 反向：仅在GPU上使用scaler
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.grad_step += 1
        if self.grad_step % self.grad_accum_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._ema_update_target()

        with torch.no_grad():
            priorities = losses.abs().cpu().numpy() + 1e-5
            self.memory.update_priorities(indices, priorities)

        return (loss.item() * self.grad_accum_steps)

    def _ema_update_target(self):
        """EMA 软更新目标网络"""
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.data.lerp_(pp.data, self.tau)

    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'grad_step': self.grad_step,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else {},
        }, path)

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint.get('steps_done', 0)
            self.grad_step = checkpoint.get('grad_step', 0)
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            for p in self.target_net.parameters():
                p.requires_grad = False
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# ============================================================
# Cosine Annealing + Linear Warmup 学习率调度
# ============================================================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr * (self.base_lrs[i] / self.base_lrs[0])

    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lrs[0] * (self.current_step / max(1, self.warmup_steps))
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state):
        self.current_step = state['current_step']
        self.base_lrs = state['base_lrs']


# ============================================================
# 批量并行训练 (使用 C++ BatchGame2048)
# ============================================================

def train_agent_batch(agent, n_envs, scheduler, episodes=100000,
                      save_path='models_v4/dqn_2048.pth',
                      checkpoint_path='models_v4/checkpoint.pth',
                      resume=False, start_episode=0):
    """
    批量并行训练 — N个环境同时采集，GPU利用率最大化
    使用 C++ BatchGame2048 实现零拷贝批量推理
    """
    if not CPP_AVAILABLE:
        raise RuntimeError("C++ engine required for batch training. "
                           "Run: pip install pybind11 && python setup.py build_ext --inplace")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_env = game2048_cpp.BatchGame2048(n_envs)
    scores_hist = []
    max_tiles_hist = []
    avg_scores = []
    losses_hist = []
    best_score = 0
    best_max_tile = 0
    total_steps = 0

    if resume and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            scores_hist = ckpt['scores']
            max_tiles_hist = ckpt['max_tiles']
            avg_scores = ckpt['avg_scores']
            losses_hist = ckpt['losses']
            best_score = ckpt.get('best_score', 0)
            best_max_tile = ckpt.get('best_max_tile', 0)
            total_steps = ckpt.get('total_steps', 0)
            scheduler.load_state_dict(ckpt['scheduler_state'])
            print(f"Resumed batch training from episode {start_episode}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            resume = False

    if not resume:
        batch_env.reset()
        start_episode = 0

    beta_start = 0.4
    total_params = sum(p.numel() for p in agent.policy_net.parameters())

    # TUI 监控 (Rich) / tqdm 回退
    monitor = None
    _live_ctx = None
    if TUI_AVAILABLE:
        from rich.live import Live
        monitor = create_monitor(
            cpp_available=CPP_AVAILABLE, compile_enabled=agent.compiled,
            n_envs=n_envs, model_params=total_params,
            config={"batch_size": agent.batch_size, "grad_accum": agent.grad_accum_steps}
        )
        if monitor:
            monitor.total_episodes = episodes
            monitor.best_score = best_score
            monitor.best_tile = best_max_tile
            _live_ctx = Live(monitor.render(), refresh_per_second=4, transient=False)
            _live_ctx.__enter__()

    progress_bar = tqdm(range(start_episode, episodes), desc="Training V4 Batch") if monitor is None else range(start_episode, episodes)

    # 预分配 tensor 避免重复分配
    states_buf = torch.empty((n_envs, 8, 4, 4), dtype=torch.float32, device=device)

    try:
     for episode in progress_bar:
        # 重置所有环境
        states_np = batch_env.reset()
        # Clear per-env n-step buffers
        for buf in agent.memory.n_step_buffer:
            buf.clear()

        ep_loss = 0.0
        ep_loss_count = 0
        total_rewards = np.zeros(n_envs, dtype=np.float32)
        env_done_count = np.zeros(n_envs, dtype=bool)
        env_scores_before = batch_env.get_scores().copy()
        step_count = 0
        max_steps = 2000  # 安全上限

        while not env_done_count.all() and step_count < max_steps:
            step_count += 1

            # --- 1. 获取有效动作掩码 ---
            valid_mask = batch_env.get_valid_masks()

            # --- 2. 批量选择动作 ---
            states_buf_np = np.asarray(states_np)
            with torch.no_grad():
                states_t = torch.from_numpy(states_buf_np).to(device, dtype=torch.float32)
                q_values = agent.policy_net(states_t)

            # 无效动作 -> -inf
            mask_t = torch.from_numpy(np.asarray(valid_mask, dtype=bool)).to(device)
            q_values_masked = q_values.clone()
            q_values_masked[~mask_t] = -float('inf')
            actions = q_values_masked.argmax(dim=1).cpu().numpy()

            # --- 3. 批量执行环境步 ---
            next_states_np, rewards_np, dones_np = batch_env.step(
                actions.astype(np.int32)
            )

            # --- 4. 存储转换 ---
            for i in range(n_envs):
                agent.memory.push(
                    states_buf_np[i],
                    int(actions[i]),
                    float(np.asarray(rewards_np)[i]),
                    np.asarray(next_states_np)[i],
                    bool(np.asarray(dones_np)[i]),
                    env_id=i
                )

            total_rewards += np.asarray(rewards_np, dtype=np.float32)

            # 标记已完成的环境
            done_arr = np.asarray(dones_np)
            env_done_count = env_done_count | done_arr

            # --- 5. 优化模型 ---
            beta = min(1.0, beta_start + (1.0 - beta_start) *
                       (episode / max(1, episodes * 0.3)))
            loss = agent.optimize_model(beta=beta)
            if loss > 0:
                ep_loss += loss
                ep_loss_count += 1
            total_steps += 1
            scheduler.step()

            # --- 6. 更新状态 ---
            states_np = next_states_np

        # 刷新 N-step buffer
        # n-step buffering handled in push() per-env with done=true flush

        # ---- 统计 ----
        final_scores = batch_env.get_scores()
        max_tiles_arr = batch_env.get_max_tiles()
        ep_score = final_scores.mean()
        ep_max_tile = max_tiles_arr.max()

        scores_hist.append(ep_score)
        max_tiles_hist.append(ep_max_tile)

        avg_loss = ep_loss / max(1, ep_loss_count)
        losses_hist.append(avg_loss)

        if ep_score > best_score:
            best_score = ep_score
            agent.save_model(save_path.replace('.pth', '_best_score.pth'))
        if ep_max_tile > best_max_tile:
            best_max_tile = ep_max_tile
            agent.save_model(save_path.replace('.pth', '_best_tile.pth'))

        recent = scores_hist[-100:] if len(scores_hist) >= 100 else scores_hist
        avg_score = np.mean(recent)
        avg_scores.append(avg_score)

        current_lr = scheduler.get_last_lr()[0]

        # 更新 TUI / tqdm
        if monitor and _live_ctx:
            monitor.update(episode + 1, episodes, ep_score, avg_score,
                          ep_max_tile, avg_loss, current_lr,
                          total_steps=total_steps,
                          best_score=best_score, best_tile=best_max_tile)
            _live_ctx.update(monitor.render())
        elif hasattr(progress_bar, 'set_description'):
            progress_bar.set_description(
                f"Ep {episode + 1} | "
                f"Score: {ep_score:.0f} (avg100: {avg_score:.0f}) | "
                f"Tile: {ep_max_tile} | "
                f"Loss: {avg_loss:.4f} | "
                f"Steps: {total_steps} | "
                f"LR: {current_lr:.2e}"
            )

        if (episode + 1) % 200 == 0:
            agent.save_model(save_path)
            checkpoint = {
                'scores': scores_hist,
                'max_tiles': max_tiles_hist,
                'avg_scores': avg_scores,
                'losses': losses_hist,
                'best_score': best_score,
                'best_max_tile': best_max_tile,
                'episode': episode + 1,
                'steps_done': agent.steps_done,
                'grad_step': agent.grad_step,
                'total_steps': total_steps,
                'scheduler_state': scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            if episode > 100:
                _plot_progress(scores_hist, avg_scores, max_tiles_hist, losses_hist)

    finally:
        if _live_ctx:
            _live_ctx.__exit__(None, None, None)

    agent.save_model(save_path)
    return scores_hist, max_tiles_hist, losses_hist


# ============================================================
# 训练函数 (单环境，兼容旧版)
# ============================================================
def train_agent(agent, env, scheduler, episodes=100000,
                save_path='models_v4/dqn_2048.pth',
                checkpoint_path='models_v4/checkpoint.pth',
                resume=False, start_episode=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scores = []
    max_tiles = []
    avg_scores = []
    losses = []
    best_score = 0
    best_max_tile = 0

    if resume and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            scores = checkpoint['scores']
            max_tiles = checkpoint['max_tiles']
            avg_scores = checkpoint['avg_scores']
            losses = checkpoint['losses']
            best_score = checkpoint.get('best_score', 0)
            best_max_tile = checkpoint.get('best_max_tile', 0)
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            print(f"Resumed from episode {start_episode}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            resume = False

    if not resume:
        start_episode = 0

    progress_bar = tqdm(range(start_episode, episodes), desc="Training V4")
    beta_start = 0.4

    for episode in progress_bar:
        state = env.reset()
        total_reward = 0.0
        done = False
        episode_loss = 0.0
        loss_count = 0

        while not done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                done = True
                continue

            action = agent.select_action(state, valid_moves)
            next_state, reward, done = env.move(action)
            total_reward += reward

            agent.memory.push(state, action, reward, next_state, done)
            state = next_state

            # 优化
            beta = min(1.0, beta_start + (1.0 - beta_start) * (episode / max(1, episodes * 0.3)))
            loss = agent.optimize_model(beta=beta)
            if loss > 0:
                episode_loss += loss
                loss_count += 1

            # 每个 step 更新 scheduler
            scheduler.step()

        # 刷新 N-step buffer 残留
        # n-step buffering handled in push() per-env with done=true flush

        score = env.score
        max_tile = int(np.max(env.board))
        scores.append(score)
        max_tiles.append(max_tile)

        avg_loss = episode_loss / max(1, loss_count)
        losses.append(avg_loss)

        if score > best_score:
            best_score = score
            agent.save_model(save_path.replace('.pth', '_best_score.pth'))
        if max_tile > best_max_tile:
            best_max_tile = max_tile
            agent.save_model(save_path.replace('.pth', '_best_tile.pth'))

        recent_scores = scores[-100:] if len(scores) >= 100 else scores
        avg_score = np.mean(recent_scores)
        avg_scores.append(avg_score)

        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_description(
            f"Ep {episode + 1} | "
            f"Score: {score} (avg100: {avg_score:.0f}) | "
            f"Tile: {max_tile} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # 每 200 轮保存
        if (episode + 1) % 200 == 0:
            agent.save_model(save_path)
            checkpoint = {
                'scores': scores,
                'max_tiles': max_tiles,
                'avg_scores': avg_scores,
                'losses': losses,
                'best_score': best_score,
                'best_max_tile': best_max_tile,
                'episode': episode + 1,
                'steps_done': agent.steps_done,
                'grad_step': agent.grad_step,
                'scheduler_state': scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            # 绘图
            if episode > 100:
                _plot_progress(scores, avg_scores, max_tiles, losses)

    agent.save_model(save_path)
    return scores, max_tiles, losses


def _plot_progress(scores, avg_scores, max_tiles, losses):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(scores, alpha=0.5, label='Score')
    plt.plot(avg_scores, label='Avg (100ep)')
    plt.xlabel('Episode'); plt.ylabel('Score')
    plt.title('Scores'); plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(max_tiles, 'g-')
    plt.xlabel('Episode'); plt.ylabel('Max Tile')
    plt.title('Max Tile')

    plt.subplot(2, 2, 3)
    plt.plot(losses, 'r-')
    plt.xlabel('Episode'); plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(2, 2, 4)
    plt.hist(scores[-500:], bins=20, alpha=0.7)
    plt.xlabel('Score'); plt.ylabel('Freq')
    plt.title('Score Dist (last 500)')

    plt.tight_layout()
    plt.savefig('training_progress_v4.png')
    plt.close()


# ============================================================
# 推理（测试）函数
# ============================================================
def play_with_model(agent, env, episodes=1):
    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        print(f"\n=== Episode {ep + 1} ===")
        while not done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                break
            action = agent.select_action(state, valid_moves, evaluate=True)
            next_state, reward, done = env.move(action)
            state = next_state
            steps += 1
        print(f"Final Score: {env.score}, Max Tile: {np.max(env.board)}, Steps: {steps}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # ---- Doctor: 自动环境检测与修复 ----
    _first_run_flag = os.path.join(os.path.dirname(__file__), ".doctor_ok")
    if not os.path.exists(_first_run_flag):
        print("First run detected — running environment diagnostics...")
        try:
            from doctor import Doctor
            d = Doctor(auto_fix=True)
            d.run()
            # Mark doctor as passed
            with open(_first_run_flag, "w") as f:
                f.write("ok")
        except ImportError:
            print("doctor.py not found, skipping diagnostics")

    # ---- 训练配置 ----
    config = {
        "train": 1,
        "resume": 1,
        "play": 0,
        "episodes": 200000,
        "lr": 3e-4,
        "gamma": 0.995,
        "tau": 0.005,
        "batch_size": 1024,
        "grad_accum_steps": 2,
        "warmup_episodes": 2000,
        "input_channels": 8,
        "action_size": 4,
        "n_envs": 64,  # 并行环境数（仅批量模式）
        "use_batch": 1,  # 1=自动(优先批量), 0=单环境
    }

    # ---- 智能体 ----
    agent = DQNAgent(
        input_channels=config["input_channels"],
        action_size=config["action_size"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        batch_size=config["batch_size"],
        grad_accum_steps=config["grad_accum_steps"],
    )

    # 打印模型参数统计
    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    trainable_params = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"torch.compile enabled: {USE_COMPILE}")
    print(f"Config: batch_size={config['batch_size']}, "
          f"grad_accum={config['grad_accum_steps']}, "
          f"effective_batch={(config['batch_size'] * config['grad_accum_steps'])}")

    # ---- 选择训练模式 ----
    use_batch = config.get("use_batch", 1) and CPP_AVAILABLE

    if use_batch:
        print(f"\n*** BATCH MODE: {config['n_envs']} parallel environments ***")
        # 批量模式: 每个 episode 的 step 数约等于平均游戏长度 (~600步)
        steps_per_ep = 600
        total_steps_estimate = config["episodes"] * steps_per_ep
    else:
        print("\n*** SINGLE-ENV MODE ***")
        env = Game2048(size=4)
        steps_per_ep = 700
        total_steps_estimate = config["episodes"] * steps_per_ep

    scheduler = CosineWarmupScheduler(
        agent.optimizer,
        warmup_steps=config["warmup_episodes"] * steps_per_ep,
        total_steps=total_steps_estimate,
        min_lr=1e-6,
    )

    # ---- 训练 ----
    if config.get("train") or config.get("resume"):
        start_episode = 0
        checkpoint_path = 'models_v4/checkpoint.pth'

        if config.get("resume") and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                start_episode = checkpoint.get('episode', 0)
                agent.load_model('models_v4/dqn_2048.pth')
                print(f"Resuming from episode {start_episode}")
            except Exception as e:
                print(f"Could not load checkpoint: {e}, starting fresh")
                start_episode = 0

        if use_batch:
            print(f"\nStarting V4 batch training ({config['n_envs']} envs) for {config['episodes']} episodes...")
            scores, max_tiles, losses = train_agent_batch(
                agent, config["n_envs"], scheduler,
                episodes=config["episodes"],
                save_path='models_v4/dqn_2048.pth',
                checkpoint_path=checkpoint_path,
                resume=config.get("resume"),
                start_episode=start_episode,
            )
        else:
            print(f"\nStarting V4 single training for {config['episodes']} episodes...")
            scores, max_tiles, losses = train_agent(
                agent, env, scheduler,
                episodes=config["episodes"],
                save_path='models_v4/dqn_2048.pth',
                checkpoint_path=checkpoint_path,
                resume=config.get("resume"),
                start_episode=start_episode,
            )
        print("Training completed!")

    # ---- 推理 ----
    if config.get("play"):
        model_path = 'models_v4/dqn_2048_best_tile.pth'
        if not os.path.exists(model_path):
            model_path = 'models_v4/dqn_2048.pth'
        if os.path.exists(model_path):
            agent.load_model(model_path)
            test_env = Game2048(size=4)
            play_with_model(agent, test_env, episodes=1)
        else:
            print("No trained model found.")
