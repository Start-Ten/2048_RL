"""Cosine annealing with linear warmup learning rate scheduler."""
import math


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
        for i, g in enumerate(self.optimizer.param_groups):
            g['lr'] = lr * (self.base_lrs[i] / self.base_lrs[0])

    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lrs[0] * self.current_step / max(1, self.warmup_steps)
        progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_last_lr(self): return [g['lr'] for g in self.optimizer.param_groups]

    def state_dict(self): return {'current_step': self.current_step, 'base_lrs': self.base_lrs}

    def load_state_dict(self, d): self.current_step = d['current_step']; self.base_lrs = d['base_lrs']
