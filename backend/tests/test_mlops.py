"""
OptiChain AI - MLOps & Experiment Tracking Unit Tests
Verifies reproducible experiment record creation, schema enforcement,
multi-seed statistics aggregation, solver diagnostics, and canonical benchmark preservation.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from optichain.mlops.experiment_tracker import (
    ExperimentTracker,
    get_git_commit,
    DEFAULT_METRIC_DEFINITIONS,
    DEFAULT_PROVENANCE_AND_LIMITATIONS,
)


@pytest.fixture
def temp_tracker(tmp_path):
    """Provides an isolated ExperimentTracker using a temporary directory."""
    return ExperimentTracker(output_dir=tmp_path / "experiments")


def test_experiment_tracker_guard_against_canonical_docs(tmp_path):
    """Tracker must raise ValueError if configured to point directly to docs/."""
    docs_dir = Path.cwd() / "docs"
    with pytest.raises(ValueError, match="cannot be set to canonical"):
        ExperimentTracker(output_dir=docs_dir)


def test_git_commit_retrieval():
    """Confirms get_git_commit returns a valid string or 'unavailable'."""
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_record_and_get_experiment(temp_tracker):
    """Tests recording an experiment and retrieving it by run_id."""
    raw_record = {
        "experiment_type": "custom_simulation",
        "policy_or_model": "Custom_ROP",
        "simulation_config": {"n_warehouses": 1, "n_products": 5},
        "aggregate_metrics": {"total_cost_mean": 120450.0},
    }
    run_id = temp_tracker.record_experiment(raw_record)
    assert run_id.startswith("exp_")

    fetched = temp_tracker.get_experiment(run_id)
    assert fetched is not None
    assert fetched["run_id"] == run_id
    assert "timestamp_utc" in fetched
    assert "git_commit" in fetched
    assert "metric_definitions" in fetched
    assert "provenance_and_limitations" in fetched
    assert fetched["aggregate_metrics"]["total_cost_mean"] == 120450.0


def test_record_multi_seed_simulation_mean_and_std(temp_tracker):
    """Verifies that multi-seed simulation calculates sample mean and std (ddof=1)."""
    seeds = [42, 101, 2024]
    per_seed = [
        {"total_cost": 100.0, "service_level": 90.0, "fill_rate": 95.0, "stockout_units": 10.0, "net_profit": 500.0},
        {"total_cost": 200.0, "service_level": 80.0, "fill_rate": 85.0, "stockout_units": 20.0, "net_profit": 400.0},
        {"total_cost": 300.0, "service_level": 70.0, "fill_rate": 75.0, "stockout_units": 30.0, "net_profit": 300.0},
    ]

    run_id = temp_tracker.record_multi_seed_simulation(
        policy_name="TestPolicy",
        seeds=seeds,
        per_seed_results=per_seed,
        simulation_config={"episode_days": 30},
    )

    exp = temp_tracker.get_experiment(run_id)
    agg = exp["aggregate_metrics"]

    # Mean = (100 + 200 + 300) / 3 = 200.0
    assert agg["total_cost_mean"] == 200.0
    # Sample std (ddof=1) = sqrt(((100-200)^2 + (200-200)^2 + (300-200)^2) / 2) = 100.0
    assert agg["total_cost_std"] == 100.0
    assert agg["service_level_mean"] == 80.0
    assert agg["service_level_std"] == 10.0


def test_record_forecasting_run(temp_tracker):
    """Verifies recording ML demand forecasting evaluation."""
    metrics = {"wape_percent": 20.62, "mae": 4.21, "rmse": 5.78, "bias_percent": -5.14}
    run_id = temp_tracker.record_forecasting_run(
        model_name="Ridge",
        horizon_days=27,
        metrics=metrics,
        test_window="Days 63..89",
        product_scope="5_SKU_Portfolio",
        run_notes="Canonical benchmark replication check",
    )

    exp = temp_tracker.get_experiment(run_id)
    assert exp["experiment_type"] == "forecasting_evaluation"
    assert exp["policy_or_model"] == "Ridge"
    assert exp["forecasting"]["horizon_days"] == 27
    assert exp["aggregate_metrics"]["wape_percent"] == 20.62


def test_record_optimization_run(temp_tracker):
    """Verifies recording MILP optimization solve with solver diagnostics."""
    run_id = temp_tracker.record_optimization_run(
        solver_name="CBC",
        solver_status="Optimal",
        is_proven_optimal=True,
        solve_time_sec=0.67,
        horizon=7,
        safety_buffer_factor=0.2,
        forecast_method="ridge",
        total_recommended_units=125.0,
        n_orders=5,
        run_notes="Production verification solve",
    )

    exp = temp_tracker.get_experiment(run_id)
    assert exp["experiment_type"] == "optimization_solve"
    assert exp["solver_diagnostics"]["solver_status"] == "Optimal"
    assert exp["solver_diagnostics"]["is_proven_optimal"] is True
    assert exp["solver_diagnostics"]["solve_time_sec"] == 0.67
    assert exp["aggregate_metrics"]["n_orders"] == 5


def test_list_and_compare_experiments(temp_tracker):
    """Verifies listing and tabular comparison of multiple experiment runs."""
    id1 = temp_tracker.record_experiment({
        "experiment_type": "policy_benchmark",
        "policy_or_model": "Policy_A",
        "aggregate_metrics": {"total_cost_mean": 1000.0, "service_level_mean": 95.0},
    })
    id2 = temp_tracker.record_experiment({
        "experiment_type": "policy_benchmark",
        "policy_or_model": "Policy_B",
        "aggregate_metrics": {"total_cost_mean": 850.0, "service_level_mean": 92.0},
    })

    runs = temp_tracker.list_experiments()
    assert len(runs) >= 2

    df = temp_tracker.compare_experiments([id1, id2])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Run ID" in df.columns
    assert "Total Cost Mean ($)" in df.columns
    assert df.loc[df["Run ID"] == id1, "Total Cost Mean ($)"].values[0] == 1000.0
    assert df.loc[df["Run ID"] == id2, "Total Cost Mean ($)"].values[0] == 850.0


def test_canonical_benchmark_preservation(temp_tracker):
    """
    Confirms that recording experiments never modifies or overwrites
    canonical benchmark artifacts in docs/.
    """
    docs_opt_file = Path("docs/optimization_benchmark_results.json")
    docs_fc_file = Path("docs/forecast_benchmark_results.json")

    assert docs_opt_file.exists()
    assert docs_fc_file.exists()

    mtime_opt_before = docs_opt_file.stat().st_mtime
    mtime_fc_before = docs_fc_file.stat().st_mtime

    temp_tracker.record_experiment({
        "experiment_type": "policy_benchmark",
        "policy_or_model": "Intrusive_Candidate",
        "aggregate_metrics": {"total_cost_mean": 999999.0},
    })

    mtime_opt_after = docs_opt_file.stat().st_mtime
    mtime_fc_after = docs_fc_file.stat().st_mtime

    assert mtime_opt_before == mtime_opt_after
    assert mtime_fc_before == mtime_fc_after
