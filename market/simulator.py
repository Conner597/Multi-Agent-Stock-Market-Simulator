from typing import Dict, List, Tuple, Type
import numpy as np

from .environment import MarketEnv
from .agents import (
    TradingAgent,
    RandomAgent,
    TrendFollowerAgent,
    ContrarianAgent,
    QLearningAgent,
)

# wrapper that wires MarkeyEnv and set of agents. Runs epiisode sims and logs results
class MarketSimulator:

    def __init__(
        self,
        agent_factories: Dict[str, TradingAgent],
        max_steps: int = 200,
        n_episodes: int = 20,
        seed: int = 42,
    ):
        self.agent_ids = list(agent_factories.keys())
        self.agent_factories = agent_factories
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed

        self._build_env_and_agents()

    def _build_env_and_agents(self):
        self.env = MarketEnv(
            agent_ids=self.agent_ids,
            max_steps=self.max_steps,
            random_seed=self.seed,
        )
        # instantiate agnets
        self.agents: Dict[str, TradingAgent] = {}
        for aid, factory in self.agent_factories.items():
            
            self.agents[aid] = factory

    def run(self) -> Dict:
        #run episodes
        price_trajectories: List[np.ndarray] = []
        episode_rewards: Dict[str, List[float]] = {aid: [] for aid in self.agent_ids}

        for ep in range(self.n_episodes):
            obs = self.env.reset()
            done = False
            t = 0
            ep_return = {aid: 0.0 for aid in self.agent_ids}

            while not done:
                # collection actions
                actions: Dict[str, int] = {}
                for aid in self.agent_ids:
                    actions[aid] = int(self.agents[aid].act(obs[aid]))

                next_obs, rewards, done, info = self.env.step(actions)
                # learning step #learning
                for aid in self.agent_ids:
                    self.agents[aid].learn(
                        obs[aid], actions[aid], rewards[aid], next_obs[aid], done
                    )
                    ep_return[aid] += rewards[aid]

                obs = next_obs
                t += 1

            price_trajectories.append(info["price_history"].copy())
            for aid in self.agent_ids:
                episode_rewards[aid].append(ep_return[aid])

        logs = {
            "price_trajectories": price_trajectories,
            "episode_rewards": episode_rewards,
            "agent_ids": self.agent_ids,
            "max_steps": self.max_steps,
            "n_episodes": self.n_episodes,
        }
        return logs
