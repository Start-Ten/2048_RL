"""Standalone model evaluation (also available via main.py test)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.agent import DQNAgent
from src.engine import Game2048


def run_eval(model_path="models_v4/dqn_2048_best_tile.pth", n_games=100):
    agent = DQNAgent()
    if not os.path.exists(model_path):
        model_path = "models_v4/dqn_2048.pth"
    if not os.path.exists(model_path):
        print(f"No model at {model_path}"); return
    agent.load(model_path)

    env = Game2048(4)
    scores, tiles = [], []
    for _ in tqdm(range(n_games), desc="Evaluating"):
        env.reset(); done = False
        while not done:
            vm = env.get_valid_moves()
            if not vm: break
            a = agent.select_action(env.get_state(), vm, evaluate=True)
            env.move(a)
        scores.append(env.score); tiles.append(env.board.max())

    os.makedirs("inference_results", exist_ok=True)
    print(f"\nAvg Score: {np.mean(scores):.0f} | Max: {max(scores)} | Min: {min(scores)}")
    print(f"Max Tile: {max(tiles)}")
    with open("inference_results/stats.txt", "w") as f:
        f.write(f"Games: {n_games}\nAvg Score: {np.mean(scores):.0f}\n")
        for t in sorted(set(tiles), reverse=True):
            f.write(f"  Tile {t}: {tiles.count(t)} games\n")

    plt.figure(figsize=(12, 8))
    plt.subplot(221); plt.hist(scores, bins=20); plt.title('Scores')
    plt.subplot(222); plt.hist(tiles, bins=20); plt.title('Max Tiles')
    plt.subplot(224); plt.scatter(tiles, scores, alpha=0.5)
    plt.tight_layout(); plt.savefig("inference_results/inference_results.png"); plt.close()
    print("Results saved to inference_results/")


if __name__ == "__main__":
    run_eval()
