"""
OptiChain AI - Experiment Tracking & Benchmark Registry
Persists model benchmarks, telemetry, and policy comparisons to disk (JSON/CSV)
with metadata for MLOps auditing and dashboard visualization.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy
from optichain.baselines.forecast_policy import MLForecastPolicy
from optichain.optimization.milp_planner import MILPReplenishmentPolicy
from optichain.rl.ppo_trainer import RLPPOPolicy


class ExperimentTracker:
    """
    Manages experiment runs, benchmark artifacts, and policy performance registries.
    """

    def __init__(self, artifacts_dir: str = "artifacts/benchmarks"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_full_benchmark(
        self,
        config: SimulationConfig,
        seeds: List[int] = [42],
        include_milp: bool = True,
        include_rl: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end multi-paradigm benchmark across classical OR heuristics,
        ML forecasting, constrained MILP, and DRL.
        """
        evaluator = PolicyEvaluator(config)
        policies = [
            MinMaxPolicy(name="Min-Max (s, S)"),
            ROPPolicy(name="Reorder Point (ROP/EOQ)"),
            PeriodicReviewPolicy(name="Periodic Review (R, S) 7-day", review_period_days=7),
            MLForecastPolicy(name="ML Forecast-Driven"),
        ]

        if include_milp:
            policies.append(MILPReplenishmentPolicy(name="Constrained MILP (PuLP)", horizon=5, time_limit_sec=2))
        if include_rl:
            policies.append(RLPPOPolicy(name="Deep RL (PPO Agent)", total_timesteps=128))

        df = evaluator.benchmark(policies, seeds=seeds)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = self.artifacts_dir / f"benchmark_results_{timestamp}.csv"
        json_path = self.artifacts_dir / "latest_benchmark.json"

        df.to_csv(csv_path, index=False)

        summary_data = {
            "timestamp": timestamp,
            "config": config.model_dump(),
            "seeds": seeds,
            "policies": df.to_dict(orient="records"),
        }

        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        return summary_data

    def load_latest_benchmark(self) -> Optional[Dict[str, Any]]:
        json_path = self.artifacts_dir / "latest_benchmark.json"
        if not json_path.exists():
            return None
        with open(json_path, "r") as f:
            return json.load(f)
