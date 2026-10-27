"""
OptiChain AI - Gymnasium RL Environment for Supply Chain Optimization
Converts DigitalTwinSimulator into a standard Farama Gymnasium environment
for training PPO, SAC, and DRL policies with continuous action spaces.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator


class SupplyChainGymEnv(gym.Env):
    """
    Gymnasium-compatible Multi-Echelon Supply Chain Digital Twin Environment.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        sim: Optional[DigitalTwinSimulator] = None,
        max_order_qty: float = 200.0,
        reward_scale: float = 1e-4,
    ):
        super().__init__()
        self.config = config or SimulationConfig()
        self.sim = sim or DigitalTwinSimulator(config=self.config)
        self.max_order_qty = max_order_qty
        self.reward_scale = reward_scale

        W = len(self.sim.warehouses)
        P = len(self.sim.products)
        self.num_actions = W * P

        # Precompute primary supplier for each (w, p) pair
        self.action_mapping: List[Tuple[int, int, str]] = []
        for w_idx in range(W):
            for p_idx in range(P):
                prod = self.sim.products[p_idx]
                eligible = [s for s in self.sim.suppliers if prod.id in s.catalog]
                if not eligible:
                    eligible = self.sim.suppliers
                best_sup = min(
                    eligible,
                    key=lambda s: s.catalog.get(prod.id, prod.unit_cost),
                )
                self.action_mapping.append((w_idx, p_idx, best_sup.id))

        # Continuous Action Space in [0.0, 1.0] for each (w, p) pair
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        # Observation Space: on-hand, pipeline, mean recent demand + calendar features
        # (W * P * 3) + 4
        self.obs_dim = (self.num_actions * 3) + 4
        self.observation_space = spaces.Box(
            low=-5.0,
            high=50.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def _extract_obs(self, sim_obs: Dict[str, Any]) -> np.ndarray:
        t = sim_obs["current_step"]
        on_hand = sim_obs["on_hand"].flatten() / 500.0
        in_transit = sim_obs["in_transit"].flatten() / 500.0

        # Demand feature: past day demand or initial mean
        t_prev = max(0, t - 1)
        recent_demand = self.sim.demand_matrix[t_prev].flatten() / 100.0

        dow = t % 7
        sin_dow = math.sin(2.0 * math.pi * dow / 7.0)
        cos_dow = math.cos(2.0 * math.pi * dow / 7.0)
        time_fraction = t / max(1, self.config.episode_days)
        disr_flag = 1.0 if sim_obs["active_disruptions"] else 0.0

        calendar_features = np.array(
            [time_fraction, sin_dow, cos_dow, disr_flag], dtype=np.float32
        )

        features = np.concatenate(
            [on_hand, in_transit, recent_demand, calendar_features]
        ).astype(np.float32)
        return features

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        sim_obs = self.sim.reset(seed=seed)
        obs = self._extract_obs(sim_obs)
        return obs, {"step": 0}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, 0.0, 1.0)
        replenishment_orders: Dict[Tuple[str, str, str], int] = {}

        for idx, (w_idx, p_idx, s_id) in enumerate(self.action_mapping):
            act_val = float(action[idx])
            qty = int(round(act_val * self.max_order_qty))
            if qty > 0:
                wh_id = self.sim.warehouses[w_idx].id
                p_id = self.sim.products[p_idx].id
                replenishment_orders[(wh_id, p_id, s_id)] = qty

        sim_obs, raw_reward, terminated, truncated, step_info = self.sim.step(
            replenishment_orders
        )
        obs = self._extract_obs(sim_obs)

        # Scaled reward formulation: maximize profit / minimize total cost
        # With bonus for achieving 100% fill rate on that step
        scaled_reward = float(step_info["net_profit"]) * self.reward_scale
        if step_info["unmet"] == 0:
            scaled_reward += 0.5  # Fill rate perfection bonus

        return obs, scaled_reward, terminated, truncated, step_info
