"""
OptiChain AI - Deep Reinforcement Learning PPO Trainer & Policy Wrapper
Trains Actor-Critic Proximal Policy Optimization (PPO) policies
and wraps them into the standardized BasePolicy benchmark interface.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from optichain.baselines.base_policy import BasePolicy
from optichain.rl.gym_env import SupplyChainGymEnv

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator


class PPOExperimentLogger(BaseCallback):
    """Custom logging callback for episode reward tracking."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        return True


class RLPPOPolicy(BasePolicy):
    """
    Trained Deep Reinforcement Learning PPO policy adapted to BasePolicy interface.
    """

    def __init__(
        self,
        name: str = "Deep RL (PPO Policy)",
        model_path: Optional[str] = None,
        total_timesteps: int = 2000,
    ):
        super().__init__(name=name)
        self.model_path = model_path
        self.total_timesteps = total_timesteps
        self.model: Optional[PPO] = None
        self.gym_env: Optional[SupplyChainGymEnv] = None

    def fit(self, sim: DigitalTwinSimulator) -> None:
        """
        Initializes the Gym environment and trains or loads the PPO agent.
        """
        self.gym_env = SupplyChainGymEnv(config=sim.config, sim=sim)

        if self.model_path and Path(self.model_path).exists():
            self.model = PPO.load(self.model_path, env=self.gym_env)
        else:
            # Train fresh PPO agent
            self.model = PPO(
                policy="MlpPolicy",
                env=self.gym_env,
                learning_rate=3e-4,
                n_steps=64,
                batch_size=32,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=0,
            )
            self.model.learn(total_timesteps=self.total_timesteps)

    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        if self.model is None or self.gym_env is None:
            return {}

        obs = self.gym_env._extract_obs(state)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(action, 0.0, 1.0)

        orders: Dict[Tuple[str, str, str], int] = {}
        for idx, (w_idx, p_idx, s_id) in enumerate(self.gym_env.action_mapping):
            act_val = float(action[idx])
            qty = int(round(act_val * self.gym_env.max_order_qty))
            if qty > 0:
                wh_id = sim.warehouses[w_idx].id
                p_id = sim.products[p_idx].id
                orders[(wh_id, p_id, s_id)] = qty

        return orders
