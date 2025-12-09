
import numpy as np
from typing import Dict, List, Tuple


class MarketEnv:

    def __init__(
        self,
        agent_ids: List[str],
        max_steps: int = 200,
        initial_price: float = 100.0,
        starting_cash: float = 1_000.0,
        starting_inventory: int = 0,
        impact_coeff: float = 0.25,
        volatility: float = 1.0,
        risk_free_rate: float = 0.0,
        random_seed: int = 42,
    ):
        self.agent_ids = list(agent_ids)
        self.max_steps = max_steps
        self.initial_price = initial_price
        self.starting_cash = starting_cash
        self.starting_inventory = starting_inventory
        self.impact_coeff = impact_coeff
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate

        self.rng = np.random.default_rng(random_seed)

        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        # reset environment to initial state
        self.t = 0
        self.price = self.initial_price
        self.price_history: List[float] = [self.price]

        self.cash = {aid: float(self.starting_cash) for aid in self.agent_ids}
        self.inventory = {aid: int(self.starting_inventory) for aid in self.agent_ids}
        self.prev_portfolio_values = {
            aid: self._portfolio_value(aid) for aid in self.agent_ids
        }

        return self._get_observations()

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        
        if set(actions.keys()) != set(self.agent_ids):
            raise ValueError("Actions must be provided for all agents.")

        a_vec = np.array([int(np.clip(actions[aid], -1, 1)) for aid in self.agent_ids])

        # compute new price
        net_demand = np.sum(a_vec)
        noise = self.rng.normal(0.0, self.volatility)
        price_change = self.impact_coeff * net_demand + noise
        self.price = max(1e-3, self.price + price_change)  
        self.price_history.append(self.price)

        # trade at new price
        for aid, action in zip(self.agent_ids, a_vec):
            # buy 1 share
            if action == 1:  
                cost = self.price
                if self.cash[aid] >= cost:
                    self.cash[aid] -= cost
                    self.inventory[aid] += 1
            elif action == -1:  
                if self.inventory[aid] > 0:
                    self.cash[aid] += self.price
                    self.inventory[aid] -= 1

        # compute rewards
        rewards: Dict[str, float] = {}
        for aid in self.agent_ids:
            pv = self._portfolio_value(aid)
            prev = self.prev_portfolio_values[aid]
            
            rewards[aid] = pv - prev
            self.prev_portfolio_values[aid] = pv

        self.t += 1
        done = self.t >= self.max_steps

        obs = self._get_observations()
        info = {
            "price": self.price,
            "price_history": np.array(self.price_history, dtype=float),
            "net_demand": float(net_demand),
        }
        return obs, rewards, done, info


    def _portfolio_value(self, agent_id: str) -> float:
        return self.cash[agent_id] + self.inventory[agent_id] * self.price

    def _get_observations(self) -> Dict[str, np.ndarray]:
        # return simple feature vectors for each agent
        prices = np.array(self.price_history, dtype=float)
        short_window = min(5, len(prices))
        long_window = min(20, len(prices))

        short_ma = float(np.mean(prices[-short_window:]))
        long_ma = float(np.mean(prices[-long_window:]))

        obs = {}
        for aid in self.agent_ids:
            inv = float(self.inventory[aid])
            norm_cash = self.cash[aid] / self.starting_cash
            obs[aid] = np.array(
                [self.price, short_ma, long_ma, inv, norm_cash], dtype=float
            )
        return obs
