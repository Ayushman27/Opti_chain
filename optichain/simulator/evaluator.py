"""
OptiChain AI - Policy Evaluation & Benchmark Harness
Provides standardized multi-seed simulation execution, comparative benchmarking,
and stress-testing against supply disruptions with fair exogenous scenario matching.
"""

from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from optichain.config import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.metrics import EpisodeKPIs
from optichain.baselines.base_policy import BasePolicy


class PolicyEvaluator:
    """
    Executes and benchmarks inventory policies under identical, fair simulation seeds.
    Keeps exogenous demand and disruption matrices matched across all competing policies.
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
        Reports per-policy total cost, fill rate, service level, stockout units,
        average inventory, and economic returns.
        """
        results: List[Dict[str, Any]] = []

        # Pre-instantiate identical simulated environments per seed to guarantee
        # that all policies face the exact same exogenous demand and disruption sequences.
        seed_environments: Dict[int, DigitalTwinSimulator] = {}
        for seed in seeds:
            sim_config = self.config.model_copy(update={"random_seed": seed})
            seed_environments[seed] = DigitalTwinSimulator(config=sim_config)

        for policy in policies:
            policy_kpis: List[EpisodeKPIs] = []

            for seed in seeds:
                # Retrieve the exact matched environment for this seed
                base_sim = seed_environments[seed]
                # Create evaluation simulator with identical entities, disruptions, and demand tensor
                eval_sim = DigitalTwinSimulator(
                    config=base_sim.config,
                    products=base_sim.products,
                    warehouses=base_sim.warehouses,
                    suppliers=base_sim.suppliers,
                    disruptions=base_sim.disruptions,
                    demand_matrix=base_sim.demand_matrix.copy(),
                )

                ep_result = self.run_episode(policy, eval_sim, seed=seed)
                policy_kpis.append(ep_result["kpis"])

            # Compute statistical averages across seeds
            fill_rates = [k.fill_rate_percent for k in policy_kpis]
            service_levels = [k.alpha_service_level_percent for k in policy_kpis]
            total_costs = [k.total_cost for k in policy_kpis]
            unmet_units = [k.total_unmet_demand for k in policy_kpis]
            avg_inventories = [k.average_on_hand_inventory for k in policy_kpis]
            revenues = [k.total_revenue for k in policy_kpis]
            profits = [k.net_profit for k in policy_kpis]
            holding_costs = [k.cost_breakdown.get("holding_cost", 0.0) for k in policy_kpis]
            stockout_costs = [k.cost_breakdown.get("stockout_penalty", 0.0) for k in policy_kpis]
            procurement_costs = [k.cost_breakdown.get("procurement_cost", 0.0) for k in policy_kpis]
            fixed_costs = [k.cost_breakdown.get("fixed_order_cost", 0.0) for k in policy_kpis]
            transport_costs = [k.cost_breakdown.get("transport_cost", 0.0) for k in policy_kpis]
            operating_costs = [k.cost_breakdown.get("operating_cost", 0.0) for k in policy_kpis]
            turnovers = [k.inventory_turnover for k in policy_kpis]

            results.append(
                {
                    "Policy": policy.name,
                    "Fill Rate (%)": round(float(np.mean(fill_rates)), 2),
                    "Service Level (%)": round(float(np.mean(service_levels)), 2),
                    "Total Cost ($)": round(float(np.mean(total_costs)), 2),
                    "Mean Total Cost ($)": round(float(np.mean(total_costs)), 2),
                    "Stockout Units": int(round(float(np.mean(unmet_units)))),
                    "Average Inventory (Units)": round(float(np.mean(avg_inventories)), 2),
                    "Holding Cost ($)": round(float(np.mean(holding_costs)), 2),
                    "Stockout Cost ($)": round(float(np.mean(stockout_costs)), 2),
                    "Procurement Cost ($)": round(float(np.mean(procurement_costs)), 2),
                    "Fixed Setup ($)": round(float(np.mean(fixed_costs)), 2),
                    "Transport ($)": round(float(np.mean(transport_costs)), 2),
                    "Facility Operating ($)": round(float(np.mean(operating_costs)), 2),
                    "Net Profit ($)": round(float(np.mean(profits)), 2),
                    "Mean Profit ($)": round(float(np.mean(profits)), 2),
                    "Inventory Turnover": round(float(np.mean(turnovers)), 2),
                }
            )


        df = pd.DataFrame(results)
        df = df.sort_values(by="Net Profit ($)", ascending=False).reset_index(drop=True)
        return df
