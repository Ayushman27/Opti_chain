"""
OptiChain AI - API Dependencies and Data Providers
Provides cached access to validated benchmark artifacts, CBC solver status,
and thread-safe, isolated simulation and forecasting services.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pulp

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.forecasting.models import ForecastEngine

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
OPTIMIZATION_BENCHMARK_PATH = DOCS_DIR / "optimization_benchmark_results.json"
FORECAST_BENCHMARK_PATH = DOCS_DIR / "forecast_benchmark_results.json"

_cached_optimization_benchmark: Optional[Dict[str, Any]] = None
_cached_forecast_benchmark: Optional[Dict[str, Any]] = None
_cached_forecast_engine: Optional[ForecastEngine] = None
_cached_sim_env: Optional[DigitalTwinSimulator] = None


def get_optimization_benchmark_path() -> Path:
    return OPTIMIZATION_BENCHMARK_PATH


def get_forecast_benchmark_path() -> Path:
    return FORECAST_BENCHMARK_PATH


def load_optimization_benchmark(force_reload: bool = False) -> Dict[str, Any]:
    """
    Loads precomputed, multi-seed MILP vs baseline benchmark results from disk.
    Cached in-memory to prevent disk thrashing.
    """
    global _cached_optimization_benchmark
    if _cached_optimization_benchmark is None or force_reload:
        if not OPTIMIZATION_BENCHMARK_PATH.exists():
            raise FileNotFoundError(
                f"Optimization benchmark artifact not found at {OPTIMIZATION_BENCHMARK_PATH}. "
                "Ensure Phase 4 benchmark scripts have generated docs/optimization_benchmark_results.json."
            )
        with open(OPTIMIZATION_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            _cached_optimization_benchmark = json.load(f)
    return _cached_optimization_benchmark


def load_forecast_benchmark(force_reload: bool = False) -> Dict[str, Any]:
    """
    Loads precomputed demand forecasting benchmark metrics from disk.
    """
    global _cached_forecast_benchmark
    if _cached_forecast_benchmark is None or force_reload:
        if not FORECAST_BENCHMARK_PATH.exists():
            raise FileNotFoundError(
                f"Forecast benchmark artifact not found at {FORECAST_BENCHMARK_PATH}. "
                "Ensure Phase 3 benchmark scripts have generated docs/forecast_benchmark_results.json."
            )
        with open(FORECAST_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            _cached_forecast_benchmark = json.load(f)
    return _cached_forecast_benchmark


def check_cbc_solver() -> bool:
    """
    Verifies that the CBC MILP solver binary is available to PuLP.
    """
    try:
        solver = pulp.PULP_CBC_CMD(msg=False)
        return bool(solver.available())
    except Exception as e:
        logger.warning("Error checking CBC solver availability: %s", e)
        return False



def get_reference_simulation(seed: int = 42) -> DigitalTwinSimulator:
    """
    Creates an isolated reference digital twin simulation for status queries
    and feature inspection.
    """
    cfg = SimulationConfig(
        n_warehouses=1,
        n_products=5,
        n_suppliers=3,
        episode_days=90,
        random_seed=seed,
    )
    return DigitalTwinSimulator(config=cfg)


def get_reference_forecast_engine(seed: int = 42) -> tuple[ForecastEngine, DigitalTwinSimulator]:
    """
    Returns an initialized forecast engine trained strictly on pre-split historical data.
    """
    global _cached_forecast_engine, _cached_sim_env
    if _cached_forecast_engine is None or _cached_sim_env is None:
        sim = get_reference_simulation(seed=seed)
        engine = ForecastEngine(random_seed=seed)
        # Train strictly on historical portion (train split)
        engine.train_and_backtest(
            demand_matrix=sim.demand_matrix,
            products=sim.products,
            warehouses=sim.warehouses,
            disruptions=sim.disruptions,
            train_split_ratio=0.70,
        )
        _cached_forecast_engine = engine
        _cached_sim_env = sim
    return _cached_forecast_engine, _cached_sim_env
