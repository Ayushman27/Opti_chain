"""
Unit and integration tests for Operations Research MILP replenishment planner.
"""

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.optimization.milp_planner import MILPReplenishmentPolicy
from optichain.simulator.evaluator import PolicyEvaluator


def test_milp_planner_action_generation():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    obs = sim.reset(seed=42)

    planner = MILPReplenishmentPolicy(horizon=3, time_limit_sec=3)
    planner.fit(sim)

    actions = planner.get_actions(obs, sim)
    assert isinstance(actions, dict)
    for (w_id, p_id, s_id), qty in actions.items():
        assert w_id == sim.warehouses[0].id
        assert qty >= 0


def test_milp_planner_episode_execution():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=4, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    planner = MILPReplenishmentPolicy(horizon=3, time_limit_sec=3)
    evaluator = PolicyEvaluator(cfg)

    result = evaluator.run_episode(planner, sim, seed=42)
    kpis = result["kpis"]

    assert kpis.total_demand > 0
    assert len(result["step_history"]) == 4
    assert kpis.fill_rate_percent >= 0.0
