import abc
import random
from typing import Dict, Tuple, Any, DefaultDict
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


Action = int  # -1, 0, 1

class TradingAgent(abc.ABC):
    # trading agents

    def __init__(self, agent_id: str, risk_aversion: float = 0.0):
        self.agent_id = agent_id
        self.risk_aversion = risk_aversion

    @abc.abstractmethod
    def act(self, obs: np.ndarray) -> Action:
        
        ...

    @abc.abstractmethod
    def learn(
        self,
        obs: np.ndarray,
        action: Action,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
     
        ...

class DQNAgent(TradingAgent):
    def __init__(self, agent_id, state_dim=5, action_dim=3, lr=1e-3, gamma=0.99, epsilon=0.1):
        super().__init__(agent_id)
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [-1, 0, 1]

        # neural network
        self.model = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, obs):
        # random exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # exploitation
        with torch.no_grad():
            q_values = self.model(torch.tensor(obs, dtype=torch.float32))
            best_index = torch.argmax(q_values).item()
        return self.actions[best_index]

    def learn(self, obs, action, reward, next_obs, done):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32)

        q_values = self.model(obs_t)
        action_index = self.actions.index(action)

        # q learning target
        with torch.no_grad():
            next_q = self.model(next_obs_t)
            target_q = reward + (0 if done else self.gamma * torch.max(next_q).item())

        
        target_vec = q_values.clone().detach()
        target_vec[action_index] = target_q

        loss = self.loss_fn(q_values, target_vec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ValueInvestorAgent(TradingAgent):
    def __init__(self, agent_id, fair_value=110):
        super().__init__(agent_id)
        self.fair_value = fair_value

    def act(self, obs):
        price = obs[0]
        if price < self.fair_value * 0.97:
            # buy undervalued
            return 1   
        elif price > self.fair_value * 1.03:
            # buy overvalued
            return -1  
        return 0

    def learn(self, *args):
        pass

class ArbitrageAgent(TradingAgent):
    def act(self, obs):
        price, short_ma, long_ma, inv, cash = obs

        # short term misprice
        if short_ma < long_ma * 0.95:
            return 1   
        elif short_ma > long_ma * 1.05:
            return -1  
        return 0

    def learn(self, *args):
        pass


class RandomAgent(TradingAgent):
    """Agent that takes random actions."""

    def act(self, obs: np.ndarray) -> Action:
        return random.choice([-1, 0, 1])

    def learn(
        self,
        obs: np.ndarray,
        action: Action,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        return


class TrendFollowerAgent(TradingAgent):
    # heuristic. Buy if price > long ma, sell if price < long ma
    def act(self, obs: np.ndarray) -> Action:
        price, short_ma, long_ma, inventory, norm_cash = obs
        if price > long_ma * 1.01:
            return 1
        elif price < long_ma * 0.99:
            return -1
        else:
            return 0

    def learn(
        self,
        obs: np.ndarray,
        action: Action,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        return


class ContrarianAgent(TradingAgent):
    # heuristic. Do opposite of current trend
    def act(self, obs: np.ndarray) -> Action:
        price, short_ma, long_ma, inventory, norm_cash = obs
        if short_ma > long_ma * 1.01:
            return -1
        elif short_ma < long_ma * 0.99:
            return 1
        else:
            return 0

    def learn(
        self,
        obs: np.ndarray,
        action: Action,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        return


class QLearningAgent(TradingAgent):
    # q learning 
    def __init__(
        self,
        agent_id: str,
        risk_aversion: float = 0.0,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        n_bins: int = 5,
        inv_range: Tuple[int, int] = (-5, 5),
    ):
        super().__init__(agent_id, risk_aversion)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_bins = n_bins
        self.inv_range = inv_range

        self.actions = [-1, 0, 1]
        self.Q: DefaultDict[Tuple[Any, ...], float] = defaultdict(float)

    def _discretize(self, obs: np.ndarray) -> Tuple[int, int, int, int, int]:
        price, short_ma, long_ma, inventory, norm_cash = obs

        def bin_scalar(x: float, lo: float, hi: float) -> int:
            x_clipped = max(lo, min(hi, x))
            rel = (x_clipped - lo) / (hi - lo + 1e-8)
            return int(rel * (self.n_bins - 1))

        price_bin = bin_scalar(price, 50.0, 150.0)
        short_bin = bin_scalar(short_ma, 50.0, 150.0)
        long_bin = bin_scalar(long_ma, 50.0, 150.0)

        inv_lo, inv_hi = self.inv_range
        inv = max(inv_lo, min(inv_hi, int(round(inventory))))
        inv_bin = int((inv - inv_lo) / (inv_hi - inv_lo + 1e-8) * (self.n_bins - 1))

        cash_bin = bin_scalar(norm_cash, 0.0, 2.0)

        return (price_bin, short_bin, long_bin, inv_bin, cash_bin)

    def _state_action_key(self, state, action: Action):
        return (*state, action)


    def act(self, obs: np.ndarray) -> Action:
        state = self._discretize(obs)
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        qs = [self.Q[self._state_action_key(state, a)] for a in self.actions]
        max_q = max(qs)
    
        best_actions = [a for a, q in zip(self.actions, qs) if q == max_q]
        return random.choice(best_actions)

    def learn(
        self,
        obs: np.ndarray,
        action: Action,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        s = self._discretize(obs)
        s_next = self._discretize(next_obs)

        key = self._state_action_key(s, action)
        q = self.Q[key]

        if done:
            target = reward
        else:
            max_next = max(
                self.Q[self._state_action_key(s_next, a)] for a in self.actions
            )
            target = reward + self.gamma * max_next

        self.Q[key] = q + self.alpha * (target - q)
