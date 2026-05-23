"""DQN Agent with SE-ResNet, NoisyNet, N-step replay, EMA target, mixed precision."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext

from .networks import DQN_V4
from .replay import NStepReplayBuffer

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda"); DEVICE_STR = "cuda"
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device("xpu"); DEVICE_STR = "xpu"
else:
    DEVICE = torch.device("cpu"); DEVICE_STR = "cpu"

# torch.compile availability
_TORCH_VER = tuple(int(x) for x in torch.__version__.split(".")[:2])
_USE_COMPILE = _TORCH_VER >= (2, 0) and DEVICE.type == "cuda"
if _USE_COMPILE:
    try:
        import triton
    except ImportError:
        _USE_COMPILE = False


# Performance optimizations
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    # Detect BF16 support (Ampere+ / compute capability >= 8.0)
    _BF16_OK = torch.cuda.is_bf16_supported()
    AMP_DTYPE = torch.bfloat16 if _BF16_OK else torch.float16
else:
    _BF16_OK = False
    AMP_DTYPE = torch.float16

class DQNAgent:
    def __init__(self, input_channels=8, action_size=4, lr=3e-4, gamma=0.995,
                 tau=0.005, batch_size=1024, grad_accum=2, n_envs=128):
        self.gamma = gamma; self.tau = tau
        self.batch_size = batch_size; self.grad_accum = grad_accum
        self.action_size = action_size

        self.policy_net = DQN_V4(input_channels, action_size).to(DEVICE)
        self.target_net = DQN_V4(input_channels, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters(): p.requires_grad = False

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = NStepReplayBuffer(capacity=200000, alpha=0.6, n_step=3,
                                        gamma=gamma, n_envs=n_envs)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self.compiled = False
        if _USE_COMPILE:
            try:
                self.policy_net = torch.compile(self.policy_net, mode="default")
                self.compiled = True
            except Exception: pass

        self.use_amp = DEVICE.type in ("cuda", "xpu")
        self.scaler = torch.amp.GradScaler(DEVICE_STR) if self.use_amp else None
        print(f"AMP: {'BF16' if (_BF16_OK and self.use_amp) else 'FP16' if self.use_amp else 'off'}")
        self.steps_done = 0; self.grad_step = 0

    def select_action(self, state, valid_moves, evaluate=False):
        self.steps_done += 1
        with torch.no_grad():
            if not evaluate: self.policy_net.reset_noise()
            s = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)
            q = self.policy_net(s).cpu().numpy().flatten()
        vq = np.full(self.action_size, -np.inf)
        for m in valid_moves: vq[m] = q[m]
        return int(np.argmax(vq))

    def optimize_model(self, beta=0.4):
        if len(self.memory) < self.batch_size: return 0.0
        s = self.memory.sample(self.batch_size, beta)
        if s is None: return 0.0
        states, actions, rewards, next_states, dones, indices, weights = s
        states = states.to(DEVICE); actions = actions.to(DEVICE)
        rewards = rewards.to(DEVICE); next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE); weights = weights.to(DEVICE)

        self.policy_net.reset_noise(); self.target_net.reset_noise()
        ctx = torch.amp.autocast(DEVICE_STR, dtype=AMP_DTYPE) if self.use_amp else nullcontext()
        with ctx:
            cur_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                na = self.policy_net(next_states).max(1)[1]
                nq = self.target_net(next_states).gather(1, na.unsqueeze(1)).squeeze()
                tgt = rewards + (1 - dones) * self.gamma * nq
            loss = ((self.loss_fn(cur_q, tgt) * weights).mean()) / self.grad_accum

        if self.use_amp: self.scaler.scale(loss).backward()
        else: loss.backward()

        self.grad_step += 1
        if self.grad_step % self.grad_accum == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            if self.use_amp:
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._ema_update()

        with torch.no_grad():
            priorities = self.loss_fn(cur_q.detach(), tgt.detach()).abs().cpu().numpy() + 1e-5
            self.memory.update_priorities(indices, priorities)
        return loss.item() * self.grad_accum

    def _ema_update(self):
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.lerp_(pp.data, self.tau)

    def save(self, path):
        d = {'policy_net': self.policy_net.state_dict(),
             'target_net': self.target_net.state_dict(),
             'optimizer': self.optimizer.state_dict(),
             'steps_done': self.steps_done, 'grad_step': self.grad_step}
        if self.scaler: d['scaler'] = self.scaler.state_dict()
        torch.save(d, path)

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy_net.load_state_dict(ck['policy_net'])
        self.target_net.load_state_dict(ck['target_net'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.steps_done = ck.get('steps_done', 0)
        self.grad_step = ck.get('grad_step', 0)
        if 'scaler' in ck and self.scaler is not None:
            self.scaler.load_state_dict(ck['scaler'])
        for p in self.target_net.parameters(): p.requires_grad = False
        return True
