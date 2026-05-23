"""N-step Prioritized Experience Replay buffer."""
import numpy as np
import torch
from collections import deque


class NStepReplayBuffer:
    def __init__(self, capacity=200000, alpha=0.6, n_step=3, gamma=0.995, n_envs=128):
        self.capacity = capacity; self.alpha = alpha
        self.n_step = n_step; self.gamma = gamma
        self.buffer = []; self.priorities = np.zeros(capacity)
        self.pos = 0; self.size = 0
        self.n_step_bufs = [deque(maxlen=n_step) for _ in range(max(1, n_envs))]

    def push(self, state, action, reward, next_state, done, env_id=0):
        buf = self.n_step_bufs[env_id]
        buf.append((state, action, reward, next_state, done))
        if done:
            while len(buf) >= 1:
                s0, a0, _, _, _ = buf[0]
                r_acc = sum((self.gamma ** t) * buf[t][2] for t in range(len(buf)))
                _, _, _, ns, d = buf[-1]
                self._store(s0, a0, r_acc, ns, d); buf.popleft()
            return
        if len(buf) < self.n_step: return
        s0, a0, _, _, _ = buf[0]
        r_acc = sum((self.gamma ** t) * buf[t][2] for t in range(self.n_step))
        _, _, _, ns, d = buf[self.n_step - 1]
        self._store(s0, a0, r_acc, ns, d); buf.popleft()

    def clear_env_buffers(self):
        for b in self.n_step_bufs: b.clear()

    def _store(self, state, action, reward, next_state, done):
        max_p = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0: return None
        p = self.priorities[:self.size]; probs = p ** self.alpha; probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.tensor(np.array(states)), torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float), torch.tensor(np.array(next_states)),
                torch.tensor(dones, dtype=torch.float), indices,
                torch.tensor(weights.astype(np.float32)))

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = max(p, 1e-6)

    def __len__(self): return self.size
