"""Training loops: single-env and batch-parallel."""
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from .agent import DEVICE, _BF16_OK
from .engine import Game2048

# Optional C++ engine
try:
    import os as _os, sys as _sys
    if _sys.platform == "win32" and hasattr(_os, 'add_dll_directory'):
        for _p in [r"C:\mingw64\bin", r"C:\Windows\mingw64\bin"]:
            if _os.path.isdir(_p): _os.add_dll_directory(_p)
    import game2048_cpp
    CPP_OK = True
except (ImportError, AttributeError):
    game2048_cpp = None
    CPP_OK = False

# Optional TUI
try:
    from ui.tui import create_monitor
    from rich.live import Live
    TUI_OK = True
except ImportError:
    create_monitor = None; Live = None
    TUI_OK = False


def _log_episode(path, ep, score, avg_score, max_tile, loss, lr):
    import csv as _csv
    existed = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = _csv.writer(f)
        if not existed: w.writerow(['episode','score','avg_score','max_tile','loss','lr'])
        w.writerow([ep, score, f'{avg_score:.0f}', max_tile, f'{loss:.6f}', f'{lr:.2e}'])

def _plot_progress(scores, avg, tiles, losses, path="training_progress.png"):
    if len(scores) < 10: return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Scores
    axes[0, 0].plot(scores, alpha=0.3, color='blue', label='Score')
    axes[0, 0].plot(avg, color='blue', linewidth=2, label='Avg 100')
    axes[0, 0].set_title('Score per Episode'); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    # Max Tile
    axes[0, 1].plot(tiles, 'g-', alpha=0.5, linewidth=1)
    axes[0, 1].set_title('Max Tile'); axes[0, 1].grid(alpha=0.3)
    # Loss
    axes[0, 2].plot(losses, 'r-', alpha=0.5, linewidth=1)
    axes[0, 2].set_title('Training Loss'); axes[0, 2].grid(alpha=0.3)
    # Score histogram
    axes[1, 0].hist(scores[-min(500, len(scores)):], bins=30, color='blue', alpha=0.7)
    axes[1, 0].set_title(f'Score Dist (last {min(500, len(scores))})')
    # Tile histogram
    tile_vals = [t for t in tiles if t >= 16]
    if tile_vals:
        axes[1, 1].hist(tile_vals, bins=20, color='green', alpha=0.7)
        axes[1, 1].set_title('Max Tile Distribution')
    # Smoothed score
    n = min(100, len(scores))
    if n > 0:
        kernel = np.ones(n)/n
        smoothed = np.convolve(scores, kernel, mode='valid')
        axes[1, 2].plot(smoothed, 'purple', linewidth=1)
        axes[1, 2].set_title(f'Score (rolling avg {n})'); axes[1, 2].grid(alpha=0.3)
    fig.suptitle(f'2048 DQN V4 — {len(scores):,} episodes', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


def train(agent, scheduler, episodes=200000, save_dir="models_v4",
          resume=False, use_batch=True, n_envs=64):
    """Main training entry point. Auto-selects batch or single-env mode."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dqn_2048.pth")
    ckpt_path = os.path.join(save_dir, "checkpoint.pth")
    csv_path = os.path.join(save_dir, "training_log.csv")

    if use_batch and not CPP_OK:
        print("C++ engine not available, falling back to single-env mode")
        use_batch = False

    if use_batch:
        return _train_batch(agent, scheduler, n_envs, episodes, save_path, ckpt_path, csv_path, resume)
    else:
        return _train_single(agent, scheduler, episodes, save_path, ckpt_path, csv_path, resume)


def _train_single(agent, scheduler, episodes, save_path, ckpt_path, csv_path, resume):
    env = Game2048(4)
    scores, max_tiles, avg_scores, losses = [], [], [], []
    best_score, best_tile = 0, 0
    start_ep = 0
    total_steps = 0

    resumed = False
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hist_path = os.path.join(os.path.dirname(save_path), 'history.pt')
        if os.path.exists(hist_path):
            hist = torch.load(hist_path, map_location=DEVICE, weights_only=False)
            scores = hist.get('scores', []); max_tiles = hist.get('max_tiles', [])
            avg_scores = hist.get('avg_scores', []); losses = hist.get('losses', [])
        else:
            scores = ck.get('scores', []); max_tiles = ck.get('max_tiles', [])
            avg_scores = ck.get('avg_scores', []); losses = ck.get('losses', [])
        best_score = ck.get('best_score', 0); best_tile = ck.get('best_max_tile', 0)
        start_ep = ck.get('episode', 0); total_steps = ck.get('total_steps', 0)
        scheduler.load_state_dict(ck['scheduler_state'])
        resumed = True
        print(f"[resume] Ep {start_ep}, best_score={best_score}, best_tile={best_tile}, "
              f"history={len(scores)} episodes")

    # TUI / tqdm
    monitor = None; live_ctx = None
    if TUI_OK:
        total_p = sum(p.numel() for p in agent.policy_net.parameters())
        monitor = create_monitor(cpp=CPP_OK, compiled=agent.compiled, n_envs=1,
                                 params=total_p,
                                 cfg={"batch_size": agent.batch_size,
                                      "grad_accum": agent.grad_accum,
                                      "amp": "BF16" if (agent.use_amp and _BF16_OK) else
                                             "FP16" if agent.use_amp else "FP32"})
        if monitor:
            monitor.total_episodes = episodes
            monitor.best_score = best_score; monitor.best_tile = best_tile
            if resumed: monitor.resumed = True; monitor.resume_ep = start_ep
            if resumed: monitor.resumed = True; monitor.resume_ep = start_ep
            live_ctx = Live(monitor.render(), refresh_per_second=4); live_ctx.__enter__()

    pbar = range(start_ep, episodes)
    if monitor is None: pbar = tqdm(pbar, desc="Training")

    _tui_last = 0.0
    try:
        for ep in pbar:
            state = env.reset(); ep_loss = 0.0; loss_n = 0
            while True:
                vm = env.get_valid_moves()
                if not vm: break
                a = agent.select_action(state, vm)
                ns, r, d = env.move(a)
                agent.memory.push(state, a, r, ns, d, env_id=0)
                state = ns; total_steps += 1; scheduler.step()
                beta = min(1.0, 0.4 + 0.6 * ep / max(1, episodes * 0.3))
                loss = agent.optimize_model(beta=beta)
                if loss > 0: ep_loss += loss; loss_n += 1
                # Periodic TUI refresh (~5s)
                if monitor and live_ctx and total_steps % 50 == 0:
                    now = time.time()
                    if now - _tui_last > 5.0:
                        _tui_last = now
                        lr = scheduler.get_last_lr()[0]
                        monitor.update(ep + 1, episodes, env.score, (avg_scores[-1] if avg_scores else 0),
                                       int(env.board.max()),
                                       ep_loss / max(1, loss_n), lr, total_steps=total_steps,
                                       best_score=best_score, best_tile=best_tile)
                        live_ctx.update(monitor.render())
                if d: break

            sc = env.score; mt = int(np.max(env.board))
            scores.append(sc); max_tiles.append(mt)
            avg_loss = ep_loss / max(1, loss_n); losses.append(avg_loss)
            if sc > best_score: best_score = sc; agent.save(save_path.replace('.pth', '_best_score.pth'))
            if mt > best_tile: best_tile = mt; agent.save(save_path.replace('.pth', '_best_tile.pth'))

            recent = scores[-100:] if len(scores) >= 100 else scores
            avg_sc = np.mean(recent); avg_scores.append(avg_sc)
            lr = scheduler.get_last_lr()[0]

            if monitor and live_ctx:
                monitor.update(ep + 1, episodes, sc, avg_sc, mt, avg_loss, lr,
                               total_steps=total_steps, best_score=best_score, best_tile=best_tile)
                live_ctx.update(monitor.render())
            elif hasattr(pbar, 'set_description'):
                pbar.set_description(f"Ep {ep+1} | Score: {sc} avg: {avg_sc:.0f} | Tile: {mt} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            # Log + checkpoint
            _log_episode(csv_path, ep + 1, sc, avg_sc, mt, avg_loss, lr)
            agent.save(save_path)
            torch.save({'episode': ep + 1, 'total_steps': total_steps,
                        'best_score': best_score, 'best_max_tile': best_tile,
                        'scheduler_state': scheduler.state_dict()}, ckpt_path)
            # Full history + plot every 200 episodes
            if (ep + 1) % 100 == 0:
                torch.save({'scores': scores, 'max_tiles': max_tiles,
                            'avg_scores': avg_scores, 'losses': losses,
                            'best_score': best_score, 'best_max_tile': best_tile,
                            'episode': ep + 1, 'total_steps': total_steps,
                            'scheduler_state': scheduler.state_dict()},
                           os.path.join(os.path.dirname(save_path), 'history.pt'))
                _plot_progress(scores, avg_scores, max_tiles, losses)
    finally:
        if live_ctx: live_ctx.__exit__(None, None, None)

    agent.save(save_path)
    return scores, max_tiles, losses


def _train_batch(agent, scheduler, n_envs, episodes, save_path, ckpt_path, csv_path, resume):
    batch_env = game2048_cpp.BatchGame2048(n_envs)
    scores, max_tiles, avg_scores, losses = [], [], [], []
    best_score, best_tile = 0, 0
    start_ep, total_steps = 0, 0

    resumed = False
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hist_path = os.path.join(os.path.dirname(save_path), 'history.pt')
        if os.path.exists(hist_path):
            hist = torch.load(hist_path, map_location=DEVICE, weights_only=False)
            scores = hist.get('scores', []); max_tiles = hist.get('max_tiles', [])
            avg_scores = hist.get('avg_scores', []); losses = hist.get('losses', [])
        else:
            scores = ck.get('scores', []); max_tiles = ck.get('max_tiles', [])
            avg_scores = ck.get('avg_scores', []); losses = ck.get('losses', [])
        best_score = ck.get('best_score', 0); best_tile = ck.get('best_max_tile', 0)
        start_ep = ck.get('episode', 0); total_steps = ck.get('total_steps', 0)
        scheduler.load_state_dict(ck['scheduler_state'])
        resumed = True
        print(f"[resume] Ep {start_ep}, best_score={best_score}, best_tile={best_tile}, "
              f"history={len(scores)} episodes")

    monitor = None; live_ctx = None
    if TUI_OK:
        total_p = sum(p.numel() for p in agent.policy_net.parameters())
        monitor = create_monitor(cpp=CPP_OK, compiled=agent.compiled, n_envs=n_envs,
                                 params=total_p,
                                 cfg={"batch_size": agent.batch_size,
                                      "grad_accum": agent.grad_accum,
                                      "amp": "BF16" if (agent.use_amp and _BF16_OK) else
                                             "FP16" if agent.use_amp else "FP32"})
        if monitor:
            monitor.total_episodes = episodes; monitor.best_score = best_score; monitor.best_tile = best_tile
            live_ctx = Live(monitor.render(), refresh_per_second=4); live_ctx.__enter__()

    pbar = range(start_ep, episodes)
    if monitor is None: pbar = tqdm(pbar, desc="Training Batch")

    _tui_last = 0.0
    try:
        for ep in pbar:
            states_np = batch_env.reset(); agent.memory.clear_env_buffers()
            ep_loss = 0.0; loss_n = 0; step_n = 0
            all_done = np.zeros(n_envs, dtype=bool)
            pushed_done = np.zeros(n_envs, dtype=bool)  # track first terminal push

            while not all_done.all() and step_n < 2000:
                step_n += 1
                mask = batch_env.get_valid_masks()
                with torch.no_grad():
                    st = torch.from_numpy(np.asarray(states_np)).to(DEVICE, dtype=torch.float32)
                    qv = agent.policy_net(st)
                mt = torch.from_numpy(np.asarray(mask, dtype=bool)).to(DEVICE)
                qv_m = qv.clone(); qv_m[~mt] = -float('inf')
                actions = qv_m.argmax(dim=1).cpu().numpy()

                ns_np, rw_np, dn_np = batch_env.step(actions.astype(np.int32))
                dn_arr = np.asarray(dn_np); all_done = all_done | dn_arr
                for i in range(n_envs):
                    # Push terminal transition ONCE, skip repeats
                    if all_done[i] and pushed_done[i]: continue
                    agent.memory.push(np.asarray(states_np)[i].copy(), int(actions[i]),
                                      float(np.asarray(rw_np)[i]),
                                      np.asarray(ns_np)[i].copy(), bool(dn_arr[i]), env_id=i)
                    if all_done[i]: pushed_done[i] = True
                beta = min(1.0, 0.4 + 0.6 * ep / max(1, episodes * 0.3))
                loss = agent.optimize_model(beta=beta)
                if loss > 0: ep_loss += loss; loss_n += 1
                total_steps += 1; scheduler.step(); states_np = ns_np
                # Periodic TUI refresh (~5s)
                if monitor and live_ctx and total_steps % 20 == 0:
                    now = time.time()
                    if now - _tui_last > 5.0:
                        _tui_last = now
                        lr = scheduler.get_last_lr()[0]
                        monitor.update(ep + 1, episodes, batch_env.get_scores().mean(),
                                       (avg_scores[-1] if avg_scores else 0),
                                       int(batch_env.get_max_tiles().max()),
                                       ep_loss / max(1, loss_n), lr, total_steps=total_steps,
                                       best_score=best_score, best_tile=best_tile)
                        live_ctx.update(monitor.render())

            sc = batch_env.get_scores().mean()
            mt = int(batch_env.get_max_tiles().max())
            scores.append(sc); max_tiles.append(mt)
            avg_loss = ep_loss / max(1, loss_n); losses.append(avg_loss)
            if sc > best_score: best_score = sc; agent.save(save_path.replace('.pth', '_best_score.pth'))
            if mt > best_tile: best_tile = mt; agent.save(save_path.replace('.pth', '_best_tile.pth'))

            recent = scores[-100:] if len(scores) >= 100 else scores
            avg_sc = np.mean(recent); avg_scores.append(avg_sc)
            lr = scheduler.get_last_lr()[0]

            if monitor and live_ctx:
                monitor.update(ep + 1, episodes, sc, avg_sc, mt, avg_loss, lr,
                               total_steps=total_steps, best_score=best_score, best_tile=best_tile)
                live_ctx.update(monitor.render())
            elif hasattr(pbar, 'set_description'):
                pbar.set_description(f"Ep {ep+1} | Score: {sc:.0f} avg: {avg_sc:.0f} | Tile: {mt} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            # Log + checkpoint
            _log_episode(csv_path, ep + 1, sc, avg_sc, mt, avg_loss, lr)
            agent.save(save_path)
            torch.save({'episode': ep + 1, 'total_steps': total_steps,
                        'best_score': best_score, 'best_max_tile': best_tile,
                        'scheduler_state': scheduler.state_dict()}, ckpt_path)
            # Full history + plot every 200 episodes
            if (ep + 1) % 100 == 0:
                torch.save({'scores': scores, 'max_tiles': max_tiles,
                            'avg_scores': avg_scores, 'losses': losses,
                            'best_score': best_score, 'best_max_tile': best_tile,
                            'episode': ep + 1, 'total_steps': total_steps,
                            'scheduler_state': scheduler.state_dict()},
                           os.path.join(os.path.dirname(save_path), 'history.pt'))
                _plot_progress(scores, avg_scores, max_tiles, losses)
    finally:
        if live_ctx: live_ctx.__exit__(None, None, None)

    agent.save(save_path)
    return scores, max_tiles, losses
