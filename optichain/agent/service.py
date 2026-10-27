"""
OptiChain AI - FastAPI REST Service
Exposes Digital Twin simulation execution, policy benchmarks, demand forecasting,
and Agentic AI reasoning endpoints for frontend dashboard integration.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy
from optichain.baselines.forecast_policy import MLForecastPolicy
from optichain.optimization.milp_planner import MILPReplenishmentPolicy
from optichain.rl.ppo_trainer import RLPPOPolicy
from optichain.mlops.experiment_tracker import ExperimentTracker
from optichain.agent.reasoning_agent import SupplyChainReasoningAgent
from optichain.forecasting.models import ForecastEngine

app = FastAPI(
    title="OptiChain AI API",
    description="Autonomous Supply Chain Optimization & Digital Twin Platform API",
    version="1.0.0",
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracker = ExperimentTracker()
reasoning_agent = SupplyChainReasoningAgent()
cached_sim: Optional[DigitalTwinSimulator] = None
cached_benchmark: Optional[Dict[str, Any]] = None


class SimulationRequest(BaseModel):
    policy: str = Field(default="min_max", description="Policy: min_max, rop, periodic, forecast, milp, ppo")
    n_warehouses: int = Field(default=3)
    n_products: int = Field(default=50)
    n_suppliers: int = Field(default=10)
    episode_days: int = Field(default=90)
    random_seed: int = Field(default=42)


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User query or instruction")
    context_kpis: Optional[Dict[str, Any]] = None


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "OptiChain AI Backend", "version": "1.0.0"}


@app.get("/api/config")
def get_default_config():
    cfg = SimulationConfig()
    return cfg.model_dump()


@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    cfg = SimulationConfig(
        n_warehouses=req.n_warehouses,
        n_products=req.n_products,
        n_suppliers=req.n_suppliers,
        episode_days=req.episode_days,
        random_seed=req.random_seed,
    )
    sim = DigitalTwinSimulator(config=cfg)

    # Select policy
    policy_map = {
        "min_max": MinMaxPolicy("Min-Max (s, S)"),
        "rop": ROPPolicy("Reorder Point (ROP/EOQ)"),
        "periodic": PeriodicReviewPolicy("Periodic Review (R, S) 7-day", review_period_days=7),
        "forecast": MLForecastPolicy("ML Forecast-Driven"),
        "milp": MILPReplenishmentPolicy("Constrained MILP (PuLP)", horizon=5, time_limit_sec=2),
        "ppo": RLPPOPolicy("Deep RL (PPO Agent)", total_timesteps=64),
    }

    selected_policy = policy_map.get(req.policy.lower(), policy_map["min_max"])
    evaluator = PolicyEvaluator(cfg)
    result = evaluator.run_episode(selected_policy, sim, seed=req.random_seed)

    disruptions_data = [d.model_dump() for d in sim.disruptions]
    diagnostics = reasoning_agent.diagnose_disruptions(result["step_history"], disruptions_data)
    cost_tradeoffs = reasoning_agent.evaluate_cost_tradeoffs(result["cost_breakdown"])

    return {
        "policy_name": result["policy_name"],
        "kpis": result["kpis"].model_dump(),
        "cost_breakdown": result["cost_breakdown"],
        "step_history": result["step_history"],
        "disruptions": disruptions_data,
        "diagnostics": diagnostics,
        "cost_tradeoffs": cost_tradeoffs,
        "warehouses": [w.model_dump() for w in sim.warehouses],
        "products_sample": [p.model_dump() for p in sim.products[:10]],
        "suppliers_sample": [s.model_dump() for s in sim.suppliers[:5]],
    }


@app.get("/api/benchmark")
def get_benchmark():
    global cached_benchmark
    if cached_benchmark is None:
        # Check disk cache
        disk_data = tracker.load_latest_benchmark()
        if disk_data is not None:
            cached_benchmark = disk_data
        else:
            cfg = SimulationConfig(n_warehouses=2, n_products=10, n_suppliers=4, episode_days=30)
            cached_benchmark = tracker.run_full_benchmark(
                cfg, seeds=[42], include_milp=True, include_rl=True
            )
    return cached_benchmark


@app.get("/api/forecast")
def get_forecast():
    cfg = SimulationConfig(n_warehouses=2, n_products=10, n_suppliers=4, episode_days=45)
    sim = DigitalTwinSimulator(config=cfg)
    engine = ForecastEngine()
    results = engine.train_and_backtest(
        sim.demand_matrix,
        sim.products,
        sim.warehouses,
        sim.disruptions,
        train_split_ratio=0.7,
    )
    return results


@app.post("/api/agent/chat")
def chat_with_agent(req: ChatRequest):
    response = reasoning_agent.generate_chat_response(
        prompt=req.prompt,
        kpis=req.context_kpis or {},
    )
    return response
