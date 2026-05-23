"""
training_tui.py -- Claude Code 风格 TUI 训练监控面板
"""
import time, math, os, sys, platform
from collections import deque

_MISSING = []
try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.console import Console, Group, RenderableType
    from rich import box
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.align import Align
except ImportError:
    _MISSING.append("rich")

try:
    import pynvml
except ImportError:
    pynvml = None

if _MISSING:
    import subprocess
    for pkg in _MISSING:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.console import Console, Group
    from rich import box
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.align import Align

# ── Palette ────────────────────────────────────────────────
C_HEADER  = "cyan"
C_ACCENT  = "bright_cyan"
C_LABEL   = "bright_black"
C_VALUE   = "white"
C_GOOD    = "green"
C_WARN    = "yellow"
C_BAD     = "red"
C_DIM     = "dim"
C_MUTED   = "bright_black"

# ── Status dots ────────────────────────────────────────────
DOT_OK   = "[green]*[/]"
DOT_WARN = "[yellow]*[/]"
DOT_OFF  = "[bright_black]-[/]"

# ── Block chars ────────────────────────────────────────────
BLOCKS = " ·▪▎▌▊█"


def _bar_ascii(value, max_val, width=16):
    """精细的条形图（半宽字符）"""
    if max_val == 0: max_val = 1
    ratio = max(0.0, min(1.0, value / max_val))
    full = int(ratio * width * 2)
    result = []
    for i in range(width):
        remain = full - i * 2
        if remain >= 2: result.append("█")
        elif remain == 1: result.append("▌")
        else: result.append(" ")
    return "".join(result)


def _gradient_bar(value, max_val, width=16):
    """渐变颜色条形图"""
    if max_val == 0: max_val = 1
    ratio = max(0.0, min(1.0, value / max_val))
    full = int(ratio * width * 2)
    colors = ["red", "yellow", "green"]
    if ratio < 0.33: c = "red"
    elif ratio < 0.66: c = "yellow"
    else: c = "green"
    result = []
    for i in range(width):
        remain = full - i * 2
        if remain >= 2: result.append(f"[{c}]█[/]")
        elif remain == 1: result.append(f"[{c}]▌[/]")
        else: result.append(f"[bright_black]·[/]")
    return "".join(result)


def _spark(data, width=28):
    """sparkline 微型趋势图"""
    if len(data) < 2:
        return "[bright_black]" + "·" * width + "[/]"
    d = list(data)
    lo, hi = min(d), max(d)
    if hi == lo: hi = lo + 1
    chars = " ▁▂▃▄▅▆▇█"
    result = []
    step = max(1, len(d) // width)
    for i in range(0, len(d), step):
        chunk = d[i:i + step]
        avg = sum(chunk) / len(chunk)
        idx = int((avg - lo) / (hi - lo) * 8) + 1
        result.append(chars[max(1, min(idx, 8))])
    return "".join(result[-width:])


def _elapsed():
    """简短的时间显示"""
    return time.strftime("%H:%M:%S")


# ── GPU Monitor ────────────────────────────────────────────
class GPUMonitor:
    def __init__(self):
        self.available = False
        self.device_count = 0
        self._handles = []
        if pynvml is None: return
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(self.device_count)]
            self.available = True
        except Exception: pass

    def get_stats(self, idx=0):
        if not self.available or idx >= self.device_count: return None
        try:
            h = self._handles[idx]
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes): name = name.decode()
            power = power_limit = temp = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception: pass
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception: pass
            return {
                "name": str(name), "util_gpu": util.gpu, "util_mem": util.memory,
                "mem_used": mem.used / 1024**3, "mem_total": mem.total / 1024**3,
                "mem_pct": (mem.used / mem.total) * 100,
                "power_w": power, "power_limit_w": power_limit, "temp_c": temp,
            }
        except Exception: return None

    def close(self):
        if self.available:
            try: pynvml.nvmlShutdown()
            except Exception: pass


# ── Training Monitor ───────────────────────────────────────
class TrainingMonitor:
    def __init__(self, cpp_available=True, compile_enabled=False,
                 n_envs=1, model_params=0, config=None):
        self.gpu = GPUMonitor()
        self.cpp = cpp_available
        self.compiled = compile_enabled
        self.n_envs = n_envs
        self.params_m = model_params / 1e6
        self.cfg = config or {}

        self._score_hist = deque(maxlen=80)
        self._loss_hist = deque(maxlen=80)
        self._tile_hist = deque(maxlen=80)
        self._gpu_hist = deque(maxlen=40)
        self._stps_hist = deque(maxlen=40)

        self.t0 = time.time()
        self._last_t = time.time()
        self._step_n = 0
        self._last_n = 0

        self.ep = 0; self.total = 0; self.score = 0; self.avg_score = 0
        self.tile = 0; self.loss = 0.0; self.lr = 0.0
        self.steps = 0; self.best_score = 0; self.best_tile = 0
        self.stps = 0.0

    def update(self, episode, total, score, avg_score, max_tile,
               loss, lr, explore=0.0, total_steps=0, best_score=0, best_tile=0,
               batch_size=None, n_envs=None):
        self.ep = episode; self.total = total
        self.score = score; self.avg_score = avg_score; self.tile = max_tile
        self.loss = loss; self.lr = lr
        self.steps = total_steps; self.best_score = best_score; self.best_tile = best_tile
        if batch_size: self.cfg["batch_size"] = batch_size
        if n_envs: self.n_envs = n_envs

        self._score_hist.append(score); self._loss_hist.append(loss)
        self._tile_hist.append(max_tile)

        self._step_n += 1
        now = time.time()
        if now - self._last_t >= 2.0:
            self.stps = (self._step_n - self._last_n) / max(0.01, now - self._last_t)
            self._last_t = now; self._last_n = self._step_n
            self._stps_hist.append(self.stps)

        gs = self.gpu.get_stats(0)
        if gs: self._gpu_hist.append(gs["util_gpu"])

    # ── Render helpers ──────────────────────────────────────

    def _label(self, text):
        return Text(text, style=C_LABEL)

    def _val(self, text, style=C_VALUE):
        return Text(str(text), style=style)

    def _row(self, label, value, style=C_VALUE):
        return [self._label(f" {label} "), self._val(value, style)]

    def _status_dot(self, condition):
        return DOT_OK if condition else DOT_OFF

    # ── Header bar ──────────────────────────────────────────
    def _header(self):
        elapsed = int(time.time() - self.t0)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

        pct = f"{self.ep / max(1, self.total) * 100:.1f}%" if self.total else "--"
        eta = ""
        if self.ep > 10 and self.total > 0 and self.stps > 0:
            steps_left = (self.total - self.ep) * 600
            eta_s = int(steps_left / self.stps)
            eh, em = eta_s // 3600, (eta_s % 3600) // 60
            eta = f"  ETA {eh}h{em:02d}m"

        left = Text()
        left.append(" 2048 DQN V4 ", style=f"bold {C_HEADER}")
        left.append(f" Ep {self.ep:,}/{self.total:,} ", style=C_VALUE)
        left.append(f"({pct}) ", style=C_DIM)
        left.append(f"T+{h}h{m:02d}m ", style=C_MUTED)
        if eta: left.append(eta, style=C_MUTED)

        right = Text()
        right.append(f" {self.stps:.1f} st/s ", style=C_MUTED)
        right.append(f" step:{self.steps:,} ", style=C_MUTED)
        right.append(f" {_elapsed()} ", style=C_DIM)

        grid = Table.grid(padding=0)
        grid.add_column(ratio=3)
        grid.add_column(ratio=1)
        grid.add_row(left, Align(right, align="right"))
        return Panel(grid, style=C_HEADER, padding=(0, 2), box=box.HEAVY)

    # ── Score card ──────────────────────────────────────────
    def _score_card(self):
        t = Table.grid(padding=(0, 2))
        t.add_column(style=C_LABEL, width=11, justify="right")
        t.add_column(style=C_VALUE)
        t.add_column(style=C_DIM, width=10)

        # Score row
        sc = f"{self.score:,}"
        best = f"best {self.best_score:,}"
        t.add_row("Score", sc, best)

        # Avg
        t.add_row("Avg 100", f"{self.avg_score:,.0f}", "")

        # Max tile (highlight if 2048+)
        tile_style = C_GOOD if self.tile >= 2048 else C_VALUE
        tile_text = f"{self.tile:,}"
        tile_best = f"best {self.best_tile:,}"
        t.add_row("Max Tile", Text(tile_text, style=tile_style), tile_best)

        # Loss
        loss_text = f"{self.loss:.4f}" if self.loss > 0 else "--"
        t.add_row("Loss", loss_text, "")

        # LR
        t.add_row("LR", f"{self.lr:.2e}", "")

        return Panel(t, title="Training", border_style="cyan", padding=(1, 2),
                     title_align="left")

    # ── System card ─────────────────────────────────────────
    def _system_card(self):
        t = Table.grid(padding=(0, 2))
        t.add_column(style=C_LABEL, width=11, justify="right")
        t.add_column(style=C_VALUE)

        t.add_row("Engine", f"{DOT_OK} C++" if self.cpp else f"{DOT_OFF} Python")
        t.add_row("Compile", f"{DOT_OK} ON" if self.compiled else f"{DOT_OFF} off")
        t.add_row("Env", f"x{self.n_envs}" if self.n_envs > 1 else "single")
        t.add_row("Params", f"{self.params_m:.0f}M")
        bs = self.cfg.get("batch_size", "--")
        ga = self.cfg.get("grad_accum", 1)
        t.add_row("Batch", f"{bs}x{ga}")

        return Panel(t, title="System", border_style="bright_black", padding=(1, 2),
                     title_align="left")

    # ── GPU card ────────────────────────────────────────────
    def _gpu_card(self):
        s = self.gpu.get_stats(0)
        if not s:
            return Panel(Align("[bright_black]GPU monitor unavailable[/]", align="center",
                              vertical="middle"),
                         title="GPU", border_style="bright_black", padding=(1, 2),
                         title_align="left")

        rows = []

        # Utilization
        u = s["util_gpu"]
        uc = C_GOOD if u > 70 else (C_WARN if u > 30 else C_BAD)
        bar = _gradient_bar(u, 100, 18)
        rows.append(Text.assemble(
            self._label(" GPU  "), bar, f" [{uc}]{u}%[/]"
        ))

        # VRAM
        m = s["mem_pct"]
        mc = C_BAD if m > 90 else C_GOOD
        bar2 = _gradient_bar(m, 100, 18)
        rows.append(Text.assemble(
            self._label(" VRAM "), bar2,
            f" [{mc}]{s['mem_used']:.1f}/{s['mem_total']:.1f} GB[/]"
        ))

        # Power
        if s.get("power_w") and s.get("power_limit_w"):
            pct = s["power_w"] / s["power_limit_w"] * 100
            pc = C_WARN if pct > 90 else C_GOOD
            bar3 = _gradient_bar(pct, 100, 18)
            rows.append(Text.assemble(
                self._label(" Power"), bar3,
                f" [{pc}]{s['power_w']:.0f}/{s['power_limit_w']:.0f}W[/]"
            ))

        # Temperature
        if s.get("temp_c") is not None:
            tc = C_BAD if s["temp_c"] > 80 else (C_WARN if s["temp_c"] > 70 else C_GOOD)
            bar4 = _gradient_bar(s["temp_c"], 90, 18)
            rows.append(Text.assemble(
                self._label(" Temp "), bar4,
                f" [{tc}]{s['temp_c']}C[/]"
            ))

        return Panel(
            Group(*rows),
            title=f"[white]{s['name']}[/]",
            border_style="bright_black", padding=(1, 2), title_align="left"
        )

    # ── Sparklines ──────────────────────────────────────────
    def _sparklines(self):
        lines = []

        # Score sparkline
        sl = _spark(self._score_hist, 44)
        lo = min(self._score_hist) if self._score_hist else 0
        hi = max(self._score_hist) if self._score_hist else 1
        lines.append(Text.assemble(
            (f" Score  ", C_LABEL),
            (f"{sl} ", C_GOOD),
            (f" {lo:,.0f} ... {hi:,.0f}", C_DIM)
        ))

        # Loss sparkline
        if self._loss_hist:
            sl2 = _spark(self._loss_hist, 44)
            lo2 = min(self._loss_hist)
            hi2 = max(self._loss_hist)
            lines.append(Text.assemble(
                (f" Loss   ", C_LABEL),
                (f"{sl2} ", C_WARN),
                (f" {lo2:.4f} ... {hi2:.4f}", C_DIM)
            ))

        # GPU util sparkline
        if self._gpu_hist:
            sl3 = _spark(self._gpu_hist, 44)
            lo3 = min(self._gpu_hist)
            hi3 = max(self._gpu_hist)
            lines.append(Text.assemble(
                (f" GPU %  ", C_LABEL),
                (f"{sl3} ", "magenta"),
                (f" {lo3:.0f} ... {hi3:.0f}%", C_DIM)
            ))

        return Panel(
            Group(*lines),
            title="Trends", border_style="bright_black", padding=(1, 2),
            title_align="left"
        )

    # ── Diagnostics ─────────────────────────────────────────
    def _diagnostics(self):
        issues = []
        suggestions = []

        gpu = self.gpu.get_stats(0)
        if gpu:
            avg_util = sum(self._gpu_hist) / max(1, len(self._gpu_hist))
            if len(self._gpu_hist) >= 8 and avg_util < 40:
                issues.append(f"{DOT_WARN} GPU util low ({avg_util:.0f}%)")
                suggestions.append(f"  -> try --n_envs {min(self.n_envs * 2, 256)}")
            if gpu["mem_pct"] > 95:
                issues.append(f"{DOT_WARN} VRAM near limit ({gpu['mem_pct']:.0f}%)")
                suggestions.append("  -> reduce batch_size or n_envs")
            if gpu.get("temp_c", 0) > 85:
                issues.append(f"{DOT_WARN} GPU hot ({gpu['temp_c']}C)")
                suggestions.append("  -> check cooling / reduce power limit")

        if not self.compiled:
            suggestions.append("  -> install triton for torch.compile (Linux CUDA)")

        if not issues:
            issues.append(f"{DOT_OK} No issues detected")
        if not suggestions:
            suggestions.append(f"  {DOT_OK} Configuration looks good")

        body = Text()
        body.append("\n".join(issues))
        body.append("\n")
        body.append(Text("\n".join(suggestions), style=C_DIM))

        return Panel(body, title="Status", border_style="bright_black",
                     padding=(1, 2), title_align="left")

    # ── Footer ──────────────────────────────────────────────
    def _footer(self):
        return Panel(
            Align(Text("Ctrl+C stop  |  R refresh  |  doctor.py --fix  for diagnostics",
                       style=C_DIM), align="center"),
            style="bright_black", padding=(0, 2)
        )

    # ── Main render ─────────────────────────────────────────
    def render(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )

        # Body: main grid
        layout["body"].split_row(
            Layout(name="left", ratio=10),
            Layout(name="right", ratio=11),
        )

        # Left column: score + sparklines
        layout["left"].split(
            Layout(name="score", size=9),
            Layout(name="spark", size=7),
            Layout(name="system", size=9),
        )

        # Right column: GPU + diagnostics
        layout["right"].split(
            Layout(name="gpu", size=11),
            Layout(name="diag"),
        )

        layout["header"].update(self._header())

        layout["score"].update(self._score_card())
        layout["spark"].update(self._sparklines())
        layout["system"].update(self._system_card())

        layout["gpu"].update(self._gpu_card())
        layout["diag"].update(self._diagnostics())

        layout["footer"].update(self._footer())

        return layout


def create_monitor(cpp_available=True, compile_enabled=False,
                   n_envs=1, model_params=0, config=None):
    try:
        return TrainingMonitor(
            cpp_available=cpp_available, compile_enabled=compile_enabled,
            n_envs=n_envs, model_params=model_params, config=config)
    except Exception as e:
        print(f"[TUI] init failed: {e}, using tqdm")
        return None
