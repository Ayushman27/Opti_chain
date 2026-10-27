"""
Unit and benchmark tests for classical heuristic policies.
"""

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy


def test_min_max_policy_execution():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=4, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = MinMaxPolicy(service_factor_z=1.65)
    evaluator = PolicyEvaluator(cfg)

    result = evaluator.run_episode(policy, sim, seed=42)
    kpis = result["kpis"]

    assert kpis.fill_rate_percent > 50.0
    assert len(result["step_history"]) == 10
    assert result["cumulative_reward"] < 0  # Costs are negative rewards


def test_rop_policy_execution():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=4, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = ROPPolicy(service_factor_z=1.65)
    evaluator = PolicyEvaluator(cfg)

    result = evaluator.run_episode(policy, sim, seed=42)
    kpis = result["kpis"]

    assert kpis.fill_rate_percent > 50.0
    assert len(result["step_history"]) == 10


def test_periodic_review_policy_execution():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=4, n_suppliers=2, episode_days=14, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = PeriodicReviewPolicy(review_period_days=5)
    evaluator = PolicyEvaluator(cfg)

    result = evaluator.run_episode(policy, sim, seed=42)
    assert len(result["step_history"]) == 14


def test_comparative_benchmark_df():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=3, n_suppliers=2, episode_days=8, random_seed=42
    )
    policies = [
        MinMaxPolicy(),
        ROPPolicy(),
        PeriodicReviewPolicy(review_period_days=4),
    ]
    evaluator = PolicyEvaluator(cfg)
    df = evaluator.benchmark(policies, seeds=[42, 99])

    assert len(df) == 3
    assert "Fill Rate (%)" in df.columns
    assert "Mean Profit ($)" in df.columns
