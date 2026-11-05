"""
OptiChain AI - Reproducible Experiment Tracking & MLOps Registry
Provides a lightweight, file-based experiment tracking system for recording
simulation benchmarks, forecasting evaluations, and MILP optimization solves.

Every run records:
- Unique run ID and UTC timestamp
- Git commit hash (or 'unavailable' fallback)
- Simulation configuration and random seed(s)
- Forecast model, horizon, data split, and formal metric definitions
- Policy and optimization parameters
- Multi-seed aggregate statistics (mean and standard deviation)
- Granular cost, service, stockout, and inventory metrics
- PuLP/CBC solver diagnostics (status, proven optimality, solve time)
- Data provenance, simulation invariants, and operational limitations

CANONICAL BENCHMARK PRESERVATION:
Experiment records are stored in artifacts/experiments/ (or a user-specified path)
and NEVER overwrite canonical benchmark artifacts in docs/.
"""

from __future__ import annotations
import json
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("optichain.mlops")

CANONICAL_BENCHMARK_FILES = {
    "optimization_benchmark_results.json",
    "forecast_benchmark_results.json",
    "optimization_audit_per_seed.json",
}

DEFAULT_METRIC_DEFINITIONS = {
    "WAPE (%)": "sum(|actual - forecast|) / sum(actual) * 100",
    "MAE": "mean(|actual - forecast|)",
    "RMSE": "sqrt(mean((actual - forecast)^2))",
    "Bias (%)": "sum(forecast - actual) / sum(actual) * 100 (positive = over-forecast, negative = under-forecast)",
    "Fill Rate (%)": "total units delivered / total units demanded * 100",
    "Cycle Service Level (%)": "periods without stockout / total periods * 100",
}

DEFAULT_PROVENANCE_AND_LIMITATIONS = {
    "data_provenance": "OptiChain AI Synthetic Multi-Echelon Retail Digital Twin",
    "temporal_split": "Train: Days 0..62 (70%), Test: Days 63..89 (30%)",
    "leakage_enforcement": "Strict rolling-origin evaluation; zero future demand observed prior to decision time",
    "inventory_conservation": "I_t = max(0, I_{t-1} + Deliveries_t - Sales_t); stockout unmet demand accounting",
    "limitations": [
        "Synthetic demand generated with additive trend, weekly seasonality, and log-normal noise",
        "Deterministic supplier lead times unless exogenous disruption is scheduled",
        "Local execution on CPU using CBC MILP branch-and-cut solver",
    ],
}


def get_git_commit() -> str:
    """Attempts to retrieve the current git HEAD commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unavailable"


class ExperimentTracker:
    """
    Lightweight, reproducible experiment tracking registry for OptiChain AI.
    Stores JSON experiment run files and maintains a JSONL registry index.
    """

    def __init__(self, output_dir: Union[str, Path] = "artifacts/experiments"):
        self.output_dir = Path(output_dir).resolve()
        
        # Guard against accidentally targeting canonical docs/ benchmark directory
        docs_dir = (Path.cwd() / "docs").resolve()
        if self.output_dir == docs_dir:
            raise ValueError(
                f"Experiment output directory cannot be set to canonical '{docs_dir}'. "
                "Canonical benchmark artifacts must never be overwritten."
            )
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.output_dir / "experiment_registry.jsonl"

    def record_experiment(self, record_data: Dict[str, Any]) -> str:
        """
        Persists a complete experiment run to disk.
        Returns the unique run_id.
        """
        # Validate or generate run ID
        run_id = record_data.get("run_id")
        if not run_id:
            now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"exp_{now_str}_{uuid.uuid4().hex[:6]}"
            record_data["run_id"] = run_id

        # Populate UTC timestamp if missing
        if "timestamp_utc" not in record_data:
            record_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        # Populate git commit if missing
        if "git_commit" not in record_data:
            record_data["git_commit"] = get_git_commit()

        # Default provenance & metric definitions
        if "metric_definitions" not in record_data:
            record_data["metric_definitions"] = DEFAULT_METRIC_DEFINITIONS
        if "provenance_and_limitations" not in record_data:
            record_data["provenance_and_limitations"] = DEFAULT_PROVENANCE_AND_LIMITATIONS

        # Save individual run file
        run_file = self.output_dir / f"run_{run_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(record_data, f, indent=2, default=str)

        # Append compact entry to registry JSONL
        compact_entry = {
            "run_id": run_id,
            "timestamp_utc": record_data["timestamp_utc"],
            "git_commit": record_data["git_commit"],
            "experiment_type": record_data.get("experiment_type", "unknown"),
            "policy_or_model": record_data.get("policy_or_model", "unknown"),
            "seeds": record_data.get("seeds", []),
            "total_cost_mean": record_data.get("aggregate_metrics", {}).get("total_cost_mean"),
            "service_level_mean": record_data.get("aggregate_metrics", {}).get("service_level_mean"),
            "file_path": str(run_file),
        }
        with open(self.registry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(compact_entry, default=str) + "\n")

        logger.info("Recorded experiment %s to %s", run_id, run_file)
        return run_id

    def get_experiment(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an experiment run record by its ID."""
        run_file = self.output_dir / f"run_{run_id}.json"
        if not run_file.exists():
            return None
        with open(run_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent experiment records from the registry."""
        if not self.registry_file.exists():
            return []
        
        runs = []
        with open(self.registry_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return runs[-limit:]

    def compare_experiments(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Loads multiple experiment records and produces a side-by-side comparison DataFrame.
        """
        rows = []
        for rid in run_ids:
            exp = self.get_experiment(rid)
            if not exp:
                continue
            agg = exp.get("aggregate_metrics", {})
            cost_bd = exp.get("cost_breakdown", {})
            solver_diag = exp.get("solver_diagnostics", {})
            
            row = {
                "Run ID": exp.get("run_id"),
                "Timestamp (UTC)": exp.get("timestamp_utc"),
                "Git Commit": exp.get("git_commit", "")[:7],
                "Type": exp.get("experiment_type"),
                "Policy / Model": exp.get("policy_or_model"),
                "Seeds": str(exp.get("seeds", [])),
                "Total Cost Mean ($)": agg.get("total_cost_mean"),
                "Total Cost Std ($)": agg.get("total_cost_std"),
                "Service Level Mean (%)": agg.get("service_level_mean"),
                "Service Level Std (%)": agg.get("service_level_std"),
                "Fill Rate Mean (%)": agg.get("fill_rate_mean"),
                "Net Profit Mean ($)": agg.get("net_profit_mean"),
                "Stockout Units Mean": agg.get("stockout_units_mean"),
                "Holding Cost ($)": cost_bd.get("holding_cost_mean"),
                "Stockout Cost ($)": cost_bd.get("stockout_cost_mean"),
                "Solver Status": solver_diag.get("solver_status"),
                "Proven Optimal": solver_diag.get("is_proven_optimal"),
                "Solve Time (s)": solver_diag.get("solve_time_sec"),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def record_forecasting_run(
        self,
        model_name: str,
        horizon_days: int,
        metrics: Dict[str, float],
        test_window: str = "Days 63..89",
        product_scope: str = "PROD_001",
        run_notes: Optional[str] = None,
    ) -> str:
        """Helper to log a machine learning demand forecasting experiment."""
        record = {
            "experiment_type": "forecasting_evaluation",
            "policy_or_model": model_name,
            "forecasting": {
                "model_name": model_name,
                "horizon_days": horizon_days,
                "product_scope": product_scope,
                "test_window": test_window,
            },
            "aggregate_metrics": metrics,
            "notes": run_notes,
        }
        return self.record_experiment(record)

    def record_optimization_run(
        self,
        solver_name: str,
        solver_status: str,
        is_proven_optimal: bool,
        solve_time_sec: float,
        horizon: int,
        safety_buffer_factor: float,
        forecast_method: str,
        total_recommended_units: float,
        n_orders: int,
        run_notes: Optional[str] = None,
    ) -> str:
        """Helper to log a MILP replenishment planning solve."""
        record = {
            "experiment_type": "optimization_solve",
            "policy_or_model": f"MILP_{forecast_method}",
            "optimization_params": {
                "horizon": horizon,
                "safety_buffer_factor": safety_buffer_factor,
                "forecast_method": forecast_method,
            },
            "solver_diagnostics": {
                "solver_name": solver_name,
                "solver_status": solver_status,
                "is_proven_optimal": is_proven_optimal,
                "solve_time_sec": solve_time_sec,
            },
            "aggregate_metrics": {
                "total_recommended_units": total_recommended_units,
                "n_orders": n_orders,
            },
            "notes": run_notes,
        }
        return self.record_experiment(record)

    def record_multi_seed_simulation(
        self,
        policy_name: str,
        seeds: List[int],
        per_seed_results: List[Dict[str, Any]],
        simulation_config: Dict[str, Any],
        run_notes: Optional[str] = None,
    ) -> str:
        """
        Helper to log a multi-seed simulation experiment, computing sample
        mean and sample standard deviation (Bessel's correction n-1) across seeds.
        """
        costs = [r["total_cost"] for r in per_seed_results]
        service_levels = [r["service_level"] for r in per_seed_results]
        fill_rates = [r["fill_rate"] for r in per_seed_results]
        stockouts = [r["stockout_units"] for r in per_seed_results]
        holding_costs = [r.get("holding_cost", 0.0) for r in per_seed_results]
        stockout_costs = [r.get("stockout_cost", 0.0) for r in per_seed_results]
        net_profits = [r.get("net_profit", 0.0) for r in per_seed_results]

        n = len(seeds)
        ddof = 1 if n > 1 else 0

        aggregate_metrics = {
            "total_cost_mean": round(float(np.mean(costs)), 2),
            "total_cost_std": round(float(np.std(costs, ddof=ddof)), 2) if n > 1 else 0.0,
            "service_level_mean": round(float(np.mean(service_levels)), 2),
            "service_level_std": round(float(np.std(service_levels, ddof=ddof)), 2) if n > 1 else 0.0,
            "fill_rate_mean": round(float(np.mean(fill_rates)), 2),
            "fill_rate_std": round(float(np.std(fill_rates, ddof=ddof)), 2) if n > 1 else 0.0,
            "stockout_units_mean": round(float(np.mean(stockouts)), 2),
            "stockout_units_std": round(float(np.std(stockouts, ddof=ddof)), 2) if n > 1 else 0.0,
            "net_profit_mean": round(float(np.mean(net_profits)), 2),
            "net_profit_std": round(float(np.std(net_profits, ddof=ddof)), 2) if n > 1 else 0.0,
        }

        cost_breakdown = {
            "holding_cost_mean": round(float(np.mean(holding_costs)), 2),
            "stockout_cost_mean": round(float(np.mean(stockout_costs)), 2),
        }

        record = {
            "experiment_type": "policy_benchmark",
            "policy_or_model": policy_name,
            "seeds": seeds,
            "simulation_config": simulation_config,
            "aggregate_metrics": aggregate_metrics,
            "cost_breakdown": cost_breakdown,
            "per_seed_results": per_seed_results,
            "notes": run_notes,
        }
        return self.record_experiment(record)
