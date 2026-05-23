#!/usr/bin/env python3
"""2048 DQN RL - Unified CLI entry point.

Usage:
  python main.py train [--episodes 200000] [--batch-size 1024] [--n-envs 64] [--resume] [--no-cpp] [--no-tui]
  python main.py play [--model PATH]
  python main.py doctor [--fix]
  python main.py web
  python main.py test [--games 100] [--model PATH]
"""
import argparse
import os
import sys
import numpy as np
import torch

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Doctor (run on first launch) ──────────────────────────
def _run_first_doctor():
    flag = os.path.join(os.path.dirname(__file__), ".doctor_ok")
    if not os.path.exists(flag):
        print("First run — running environment diagnostics...")
        try:
            from tools.doctor import Doctor
            Doctor(auto_fix=True).run()
            with open(flag, "w") as f: f.write("ok")
        except ImportError:
            print("doctor not available, skipping")


# ── Subcommands ───────────────────────────────────────────

def cmd_train(args):
    from src.agent import DQNAgent, DEVICE
    from src.scheduler import CosineWarmupScheduler
    from src.trainer import train

    print(f"Device: {DEVICE}")
    print(f"Config: episodes={args.episodes} batch={args.batch_size} n_envs={args.n_envs}")

    agent = DQNAgent(
        input_channels=8, action_size=4,
        lr=args.lr, gamma=args.gamma, tau=args.tau,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        n_envs=args.n_envs)

    total_p = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"Model: {total_p/1e6:.1f}M params, "
          f"compile={'ON' if agent.compiled else 'off'}, "
          f"amp={'ON' if agent.use_amp else 'off'}")

    steps_per_ep = 600 if args.use_batch else 700
    total_steps = args.episodes * steps_per_ep
    scheduler = CosineWarmupScheduler(
        agent.optimizer, warmup_steps=2000 * steps_per_ep,
        total_steps=total_steps, min_lr=1e-6)

    # Resume
    start_ep = 0
    ckpt_path = os.path.join(args.save_dir, "checkpoint.pth")
    model_path = os.path.join(args.save_dir, "dqn_2048.pth")
    if args.resume and os.path.exists(model_path):
        try:
            agent.load(model_path)
            ck = torch.load(ckpt_path, map_location=agent.policy_net.device if False else 'cpu',
                            weights_only=False)
            start_ep = ck.get('episode', 0)
            print(f"Resumed from episode {start_ep}")
        except Exception as e:
            print(f"Could not resume: {e}"); start_ep = 0

    print(f"Starting {'batch' if args.use_batch else 'single-env'} training "
          f"for {args.episodes} episodes...")
    train(agent, scheduler, episodes=args.episodes, save_dir=args.save_dir,
          resume=args.resume, use_batch=args.use_batch, n_envs=args.n_envs)
    print("Training completed.")


def cmd_play(args):
    from src.agent import DQNAgent
    from src.engine import Game2048

    agent = DQNAgent()
    model_path = args.model or "models_v4/dqn_2048_best_tile.pth"
    if not os.path.exists(model_path):
        model_path = "models_v4/dqn_2048.pth"
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        sys.exit(1)

    agent.load(model_path)
    env = Game2048(4)
    state = env.reset(); done = False; steps = 0
    print(f"Playing with model: {model_path}")
    print(env.board)

    while not done:
        vm = env.get_valid_moves()
        if not vm: break
        action = agent.select_action(state, vm, evaluate=True)
        state, reward, done = env.move(action); steps += 1
        print(f"\nStep {steps}: {['Up','Right','Down','Left'][action]}")
        print(env.board)
        print(f"Score: {env.score}  Max: {env.board.max()}")

    print(f"\nGame Over! Score: {env.score}  Max Tile: {env.board.max()}  Steps: {steps}")


def cmd_doctor(args):
    from tools.doctor import Doctor
    Doctor(auto_fix=args.fix).run()


def cmd_web(args):
    from web.app import demo
    demo.launch()


def cmd_test(args):
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from src.agent import DQNAgent, DEVICE
    from src.engine import Game2048

    agent = DQNAgent()
    model_path = args.model or "models_v4/dqn_2048_best_tile.pth"
    if not os.path.exists(model_path):
        model_path = "models_v4/dqn_2048.pth"
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        sys.exit(1)

    agent.load(model_path)
    env = Game2048(4)
    scores, tiles, steps_list = [], [], []

    print(f"Running {args.games} evaluation games...")
    for _ in tqdm(range(args.games)):
        state = env.reset(); done = False; steps = 0
        while not done:
            vm = env.get_valid_moves()
            if not vm: break
            action = agent.select_action(state, vm, evaluate=True)
            state, _, done = env.move(action); steps += 1
        scores.append(env.score); tiles.append(env.board.max()); steps_list.append(steps)

    os.makedirs("inference_results", exist_ok=True)
    print(f"\nAvg Score: {np.mean(scores):.0f}  Median: {np.median(scores):.0f}  "
          f"Max: {max(scores)}  Min: {min(scores)}")
    print(f"Avg Tile: {np.mean(tiles):.0f}  Most Common: {max(set(tiles), key=tiles.count)}")
    print(f"Max Tile: {max(tiles)}  Min Tile: {min(tiles)}")
    print(f"Avg Steps: {np.mean(steps_list):.0f}")

    with open("inference_results/stats.txt", "w") as f:
        f.write(f"Games: {args.games}\nAvg Score: {np.mean(scores):.0f}\n"
                f"Max Tile: {max(tiles)}\n")
        for t in sorted(set(tiles), reverse=True):
            f.write(f"  Tile {t}: {tiles.count(t)} games\n")

    plt.figure(figsize=(12, 8))
    plt.subplot(221); plt.hist(scores, bins=20); plt.title('Scores')
    plt.subplot(222); plt.hist(tiles, bins=20); plt.title('Max Tiles')
    plt.subplot(223); plt.hist(steps_list, bins=20); plt.title('Steps')
    plt.subplot(224); plt.scatter(tiles, scores, alpha=0.5); plt.title('Score vs Tile')
    plt.tight_layout(); plt.savefig("inference_results/inference_results.png"); plt.close()
    print("Results saved to inference_results/")


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="2048 DQN RL Training & Inference")
    sub = parser.add_subparsers(dest="command")

    # train
    p = sub.add_parser("train", help="Train the DQN agent")
    p.add_argument("--episodes", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-cpp", dest="use_batch", action="store_false", default=True)
    p.add_argument("--no-tui", dest="no_tui", action="store_true")
    p.add_argument("--save-dir", default="models_v4")

    # play
    p2 = sub.add_parser("play", help="Play a game with trained model")
    p2.add_argument("--model", help="Path to model checkpoint")

    # doctor
    p3 = sub.add_parser("doctor", help="Environment diagnostics")
    p3.add_argument("--fix", action="store_true", help="Auto-fix issues")

    # web
    sub.add_parser("web", help="Launch Gradio web interface")

    # test
    p5 = sub.add_parser("test", help="Evaluate trained model")
    p5.add_argument("--games", type=int, default=100)
    p5.add_argument("--model", help="Path to model checkpoint")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    _run_first_doctor()

    cmds = {"train": cmd_train, "play": cmd_play, "doctor": cmd_doctor,
            "web": cmd_web, "test": cmd_test}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
