"""
OptiChain AI - Policy Evaluation & Benchmark Harness
Provides standardized multi-seed simulation execution, comparative benchmarking,
and stress-testing against supply disruptions.
"""

from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.metrics import EpisodeKPIs
from optichain.baselines.base_policy import BasePolicy


class PolicyEvaluator:
    """
    Executes and benchmarks inventory policies under identical, fair simulation seeds.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run_episode(
        self,
        policy: BasePolicy,
        sim: DigitalTwinSimulator,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs a complete episode for a single policy.
        """
        # Fit policy on simulator's network and demand parameters
        policy.fit(sim)
        obs = sim.reset(seed=seed)
        terminated = False
        cumulative_reward = 0.0

        while not terminated:
            actions = policy.get_actions(obs, sim)
            obs, reward, terminated, _, _ = sim.step(actions)
            cumulative_reward += reward

        kpis = sim.get_kpis()
        return {
            "policy_name": policy.name,
            "seed": seed,
            "kpis": kpis,
            "cumulative_reward": cumulative_reward,
            "step_history": list(sim.step_history),
            "cost_breakdown": sim.cost_engine.get_summary(),
        }

    def benchmark(
        self,
        policies: List[BasePolicy],
        seeds: List[int] = [42, 101, 777],
    ) -> pd.DataFrame:
        """
        Runs fair multi-seed evaluation across all candidate policies.
        Returns a comparative summary DataFrame.
        """
        results: List[Dict[str, Any]] = []

        for policy in policies:
            policy_kpis: List[EpisodeKPIs] = []

            for seed in seeds:
                # Fresh simulator per seed to ensure identical conditions across policies
                sim_config = self.config.model_copy(update={"random_seed": seed})
                sim = DigitalTwinSimulator(config=sim_config)

                ep_result = self.run_episode(policy, sim, seed=seed)
                policy_kpis.append(ep_result["kpis"])

            # Compute means across seeds
            fill_rates = [k.fill_rate_percent for k in policy_kpis]
            service_levels = [k.alpha_service_level_percent for k in policy_kpis]
            total_costs = [k.total_cost for k in policy_kpis]
            revenues = [k.total_revenue for k in policy_kpis]
            profits = [k.net_profit for k in policy_kpis]
            holding_costs = [k.cost_breakdown.get("holding_cost", 0.0) for k in policy_kpis]
            stockout_costs = [k.cost_breakdown.get("stockout_penalty", 0.0) for k in policy_kpis]
            turnovers = [k.inventory_turnover for k in policy_kpis]

            results.append(
                {
                    "Policy": policy.name,
                    "Fill Rate (%)": round(float(np.mean(fill_rates)), 2),
                    "Service Level (%)": round(float(np.mean(service_levels)), 2),
                    "Mean Total Cost ($)": round(float(np.mean(total_costs)), 2),
                    "Mean Revenue ($)": round(float(np.mean(revenues)), 2),
                    "Mean Profit ($)": round(float(np.mean(profits)), 2),
                    "Holding Cost ($)": round(float(np.mean(holding_costs)), 2),
                    "Stockout Cost ($)": round(float(np.mean(stockout_costs)), 2),
                    "Inventory Turnover": round(float(np.mean(turnovers)), 2),
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values(by="Mean Profit ($)", ascending=False).reset_index(drop=True)
        return df
