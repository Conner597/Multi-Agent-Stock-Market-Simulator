import argparse
import json
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from market.agents import (
    RandomAgent,
    TrendFollowerAgent,
    ContrarianAgent,
    QLearningAgent,
    DQNAgent,
    ValueInvestorAgent,
    ArbitrageAgent,
)
from market.simulator import MarketSimulator

def interactive_setup():
    print("Welcome to the Market Simulator Setup!")

    episodes = int(input("How many episodes? "))
    max_steps = int(input("How many steps per episode? "))
    seed = int(input("Random seed: "))

    return episodes, max_steps, seed


def build_demo_scenario():
    # makes market with agent
    agents = {
        "deep_rl": DQNAgent("deep_rl"),
        "value": ValueInvestorAgent("value"),
        "arb": ArbitrageAgent("arb"),
        "ql_trader": QLearningAgent("ql_trader", risk_aversion=0.01),
        "trend": TrendFollowerAgent("trend", risk_aversion=0.02),
        "contrarian": ContrarianAgent("contrarian", risk_aversion=0.02),
        "noise": RandomAgent("noise"),
    }
    return agents


def run_experiment(args):
    if args.scenario == "demo":
        agents = build_demo_scenario()
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    sim = MarketSimulator(
        agent_factories=agents,
        max_steps=args.max_steps,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    logs = sim.run()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # save logs as JSONs
    serializable_logs = {
        "price_trajectories": [traj.tolist() for traj in logs["price_trajectories"]],
        "episode_rewards": logs["episode_rewards"],
        "agent_ids": logs["agent_ids"],
        "max_steps": logs["max_steps"],
        "n_episodes": logs["n_episodes"],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"experiment_logs_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(serializable_logs, f, indent=2)

    print(f"Saved experiment logs to: {json_path}")

    if args.plot:
        plot_results(serializable_logs, out_dir, timestamp)


def plot_results(logs, out_dir, tag):
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
    price_fig_path = os.path.join(out_dir, f"prices_{tag}.png")
    plt.savefig(price_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved price plot to: {price_fig_path}")

    # episode rewards per agent
    plt.figure()
    xs = np.arange(1, logs["n_episodes"] + 1)
    for aid in agent_ids:
        plt.plot(xs, episode_rewards[aid], marker="o", label=aid)
    plt.xlabel("Episode")
    plt.ylabel("Total reward (change in portfolio value)")
    plt.title("Episode returns per agent")
    plt.legend()
    rew_fig_path = os.path.join(out_dir, f"episode_returns_{tag}.png")
    plt.savefig(rew_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved rewards plot to: {rew_fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulated Financial Markets as Multi-Agent RL Environments"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="demo",
        help="Which predefined scenario to run (default: demo)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to simulate (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Where to save logs and plots (default: ./results)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate matplotlib plots after running.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    episodes, max_steps, seed = interactive_setup()
    args = argparse.Namespace(
        scenario="demo",
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        output_dir="results",
        plot=True
    )
    run_experiment(args)

