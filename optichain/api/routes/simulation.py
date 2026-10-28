"""
Simulation run route
POST /api/v1/simulation/run
"""

import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from optichain.api.schemas import SimulationRunRequest, SimulationRunResponse
from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy
from optichain.baselines.forecast_policy import MLForecastPolicy
from optichain.optimization.milp_planner import MILPReplenishmentPolicy

router = APIRouter(prefix="/simulation", tags=["Simulation"])


@router.post("/run", response_model=SimulationRunResponse)
def run_simulation(req: SimulationRunRequest) -> SimulationRunResponse:
    """
    Executes a controlled, synchronous digital twin simulation run for a chosen policy
    under user-defined network scale, horizon duration, and RNG seed.
    Guarded with parameter validation and execution bounds.
    """
    cfg = SimulationConfig(
        n_warehouses=req.n_warehouses,
        n_products=req.n_products,
        n_suppliers=req.n_suppliers,
        episode_days=req.episode_days,
        random_seed=req.random_seed,
    )
    sim = DigitalTwinSimulator(config=cfg)

    policy_map = {
        "min_max": MinMaxPolicy("Min-Max (s, S) Policy"),
        "rop": ROPPolicy("ROP Order-Up-To Target"),
        "periodic": PeriodicReviewPolicy("Periodic Review (R, S) 7-day", review_period_days=7),
        "forecast": MLForecastPolicy("ML Forecast-Driven"),
        "milp": MILPReplenishmentPolicy("Forecast-Driven MILP Optimizer", horizon=5, time_limit_sec=2.0),
    }

    selected_policy = policy_map.get(req.policy)
    if not selected_policy:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported policy '{req.policy}'. Supported policies: {list(policy_map.keys())}"
        )

    try:
        evaluator = PolicyEvaluator(cfg)
        result = evaluator.run_episode(selected_policy, sim, seed=req.random_seed)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation run failed: {str(e)}"
        )

    run_id = f"sim_{uuid.uuid4().hex[:12]}"
    raw_history = result.get("step_history", [])

    # Sample step history evenly (at most 20 points) for lightweight client transmission
    step_count = len(raw_history)
    sample_interval = max(1, step_count // 20)
    sampled_history = [
        raw_history[i] for i in range(0, step_count, sample_interval)
    ]
    if raw_history and (step_count - 1) % sample_interval != 0:
        sampled_history.append(raw_history[-1])

    kpis_dict = result["kpis"].model_dump()
    cost_dict = result["cost_breakdown"]

    return SimulationRunResponse(
        run_id=run_id,
        policy_name=result["policy_name"],
        episode_days=req.episode_days,
        seed=req.random_seed,
        kpis=kpis_dict,
        cost_breakdown=cost_dict,
        step_history_sample=sampled_history,
        total_steps=step_count,
        disruptions_count=len(sim.disruptions),
    )
