"""
Unit tests for Gymnasium environment and PPO reinforcement learning policy.
"""

import numpy as np
from optichain.core.entities import SimulationConfig
from optichain.rl.gym_env import SupplyChainGymEnv
from optichain.rl.ppo_trainer import RLPPOPolicy
from optichain.simulator.environment import DigitalTwinSimulator


def test_gym_env_spaces_and_reset():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=3, n_suppliers=2, episode_days=10, random_seed=42
    )
    env = SupplyChainGymEnv(config=cfg)
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (2 * 3,)
    assert info["step"] == 0


def test_gym_env_step_transition():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=5, random_seed=42
    )
    env = SupplyChainGymEnv(config=cfg)
    obs, _ = env.reset(seed=42)

    action = np.array([0.5, 0.2], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert not terminated
    assert info["step"] == 0


def test_rl_ppo_policy_fit_and_action():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = RLPPOPolicy(total_timesteps=64)  # Rapid test timesteps
    policy.fit(sim)

    obs = sim.reset(seed=42)
    actions = policy.get_actions(obs, sim)
    assert isinstance(actions, dict)
