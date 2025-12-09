
import json
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_and_plot(log_path: str, output_dir: str = None):
    with open(log_path, "r") as f:
        logs = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(log_path)

    price_trajectories = logs["price_trajectories"]
    agent_ids = logs["agent_ids"]
    episode_rewards = logs["episode_rewards"]

    # price trajectories
    plt.figure()
    for i, traj in enumerate(price_trajectories):
        plt.plot(traj, alpha=0.7, label=f"Episode {i+1}")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.title("Price trajectories across episodes")
    plt.legend()
    price_fig_path = os.path.join(output_dir, "prices_from_log.png")
    plt.savefig(price_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved price plot to: {price_fig_path}")

    # episode returns per agent
    plt.figure()
    xs = np.arange(1, logs["n_episodes"] + 1)
    for aid in agent_ids:
        plt.plot(xs, episode_rewards[aid], marker="o", label=aid)
    plt.xlabel("Episode")
    plt.ylabel("Total reward (change in portfolio value)")
    plt.title("Episode returns per agent")
    plt.legend()
    rew_fig_path = os.path.join(output_dir, "episode_returns_from_log.png")
    plt.savefig(rew_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved rewards plot to: {rew_fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize saved experiment logs.")
    parser.add_argument("log_path", type=str, help="Path to experiment_logs_*.json")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as log file).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_and_plot(args.log_path, args.output_dir)
