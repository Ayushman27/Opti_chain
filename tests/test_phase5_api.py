"""
Tests for OptiChain AI Phase 5 FastAPI REST Service
Validates all /api/v1 endpoints: health, dashboard summary, benchmarks,
forecasting, inventory, MILP optimization recommendations, and simulation execution.
"""

from fastapi.testclient import TestClient
from optichain.api.main import app
import pytest

client = TestClient(app)


# -------------------------------------------------------------------------
# 1. Health Endpoint Tests
# -------------------------------------------------------------------------

def test_v1_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["version"] == "1.0.0"
    assert data["service"] == "OptiChain AI Backend"
    assert isinstance(data["cbc_solver_available"], bool)
    assert data["benchmark_artifact_present"] is True
    assert "timestamp" in data


def test_root_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == "1.0.0"


# -------------------------------------------------------------------------
# 2. Dashboard Summary Endpoint Tests
# -------------------------------------------------------------------------

def test_v1_dashboard_summary():
    res = client.get("/api/v1/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "data_source" in data
    assert "docs/optimization_benchmark_results.json" in data["data_source"]
    assert len(data["policies"]) == 4
    assert "Forecast-Driven MILP Optimizer" in data["optimal_policy"]
    assert "ROP Order-Up-To Target" in data["baseline_policy"]
    
    # Assert numerical consistency with Phase 4 validated benchmarks
    assert abs(data["total_cost"] - 428534.15) < 1.0
    assert abs(data["net_profit"] - 98124.82) < 1.0
    assert abs(data["fill_rate_percent"] - 98.65) < 0.1
    assert abs(data["service_level_percent"] - 89.63) < 0.1
    assert data["stockout_units"] == 132.0
    assert abs(data["average_inventory_units"] - 486.65) < 0.1
    assert abs(data["inventory_turnover"] - 18.82) < 0.1

    # Check relative savings vs ROP
    assert data["cost_savings_vs_rop_percent"] > 0
    assert data["net_profit_uplift_percent"] > 0


# -------------------------------------------------------------------------
# 3. Benchmarks Endpoint Tests
# -------------------------------------------------------------------------

def test_v1_benchmarks():
    res = client.get("/api/v1/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert "configuration" in data
    assert data["configuration"]["seeds"] == [42, 101, 777]
    assert len(data["policies"]) == 4

    # Verify all 4 policies are present
    policy_names = [p["policy"] for p in data["policies"]]
    assert any("MILP" in p for p in policy_names)
    assert any("Periodic" in p for p in policy_names)
    assert any("Min-Max" in p for p in policy_names)
    assert any("ROP" in p for p in policy_names)

    # Check that each policy includes cost components and stats
    for p in data["policies"]:
        assert p["holding_cost_usd"] >= 0
        assert p["stockout_cost_usd"] >= 0
        assert p["procurement_cost_usd"] >= 0
        assert p["total_cost_usd"] > 0
        assert p["net_profit_usd"] > 0
        if p["stats"]:
            assert "total_cost" in p["stats"]
            assert p["stats"]["total_cost"]["mean"] > 0

    # Verify solver diagnostics
    diag = data["solver_diagnostics"]
    assert diag is not None
    assert diag["total_solves"] == 270
    assert diag["proven_optimal_solves"] == 264
    assert diag["feasible_incumbent_solves"] == 6
    assert diag["time_limited_solves"] == 6
    assert diag["infeasible_solves"] == 0


# -------------------------------------------------------------------------
# 4. Forecasts Endpoint Tests
# -------------------------------------------------------------------------

def test_v1_forecasts_default():
    res = client.get("/api/v1/forecasts")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "Ridge"
    assert data["product_id"] == "PROD_001"
    assert data["horizon_days"] == 14
    assert len(data["historical_steps"]) > 0
    assert len(data["historical_actual"]) == len(data["historical_steps"])
    assert len(data["evaluation_steps"]) == 14
    assert len(data["evaluation_actual"]) == 14
    assert len(data["evaluation_predictions"]) == 14
    assert len(data["future_steps"]) == 14
    assert len(data["future_predictions"]) == 14
    assert "metrics" in data
    assert data["metrics"]["wape_percent"] >= 0
    assert data["metrics"]["mae"] >= 0
    assert "leakage_audit_status" in data


def test_v1_forecasts_lightgbm_custom_horizon():
    res = client.get("/api/v1/forecasts?product_id=PROD_002&model=LightGBM&horizon=7")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "LightGBM"
    assert data["product_id"] == "PROD_002"
    assert data["horizon_days"] == 7
    assert len(data["evaluation_steps"]) == 7


def test_v1_forecasts_invalid_product():
    res = client.get("/api/v1/forecasts?product_id=PROD_UNKNOWN")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_v1_forecasts_invalid_model():
    res = client.get("/api/v1/forecasts?model=InvalidNeuralNet")
    assert res.status_code == 422
    assert "not supported" in res.json()["detail"].lower()


def test_v1_forecasts_horizon_out_of_bounds():
    res = client.get("/api/v1/forecasts?horizon=50")
    assert res.status_code == 422


# -------------------------------------------------------------------------
# 5. Inventory Endpoint Tests
# -------------------------------------------------------------------------

def test_v1_inventory():
    res = client.get("/api/v1/inventory")
    assert res.status_code == 200
    data = res.json()
    assert data["as_of_step"] == 14
    assert data["total_on_hand_units"] > 0
    assert data["total_fulfilled_demand"] > 0
    assert data["overall_fill_rate_percent"] >= 80.0
    assert len(data["products"]) == 5

    for prod in data["products"]:
        assert prod["product_id"].startswith("PROD_")
        assert prod["on_hand_units"] >= 0
        assert prod["pipeline_units"] >= 0
        assert prod["reorder_point"] > 0
        assert prod["safety_stock"] > 0
        assert prod["stock_status"] in ("NORMAL", "LOW_STOCK", "CRITICAL_STOCKOUT", "OVERSTOCKED")
        assert prod["recommendation"] is not None


# -------------------------------------------------------------------------
# 6. Optimization Recommendations Endpoint Tests
# -------------------------------------------------------------------------

def test_v1_optimization_recommendations_success():
    payload = {
        "horizon": 5,
        "safety_buffer_factor": 0.2,
        "time_limit_sec": 4,
        "forecast_method": "ridge",
    }
    res = client.post("/api/v1/optimization/recommendations", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["solver_status"] in ("Optimal", "IntegerFeasible")
    assert data["termination_reason"] in ("ProvenOptimal", "FeasibleIncumbentUnproven")
    assert isinstance(data["is_proven_optimal"], bool)
    assert data["is_feasible"] is True
    assert data["solve_time_sec"] > 0
    assert data["planning_horizon_days"] == 5
    assert data["forecast_method_used"] == "ridge"
    assert isinstance(data["recommended_orders"], list)
    assert "READ-ONLY DECISION REVIEW" in data["decision_review_notice"]

    # Verify orders structure if any recommendations generated
    for order in data["recommended_orders"]:
        assert order["quantity"] > 0
        assert order["product_id"].startswith("PROD_")
        assert order["warehouse_id"].startswith("WH_")
        assert order["supplier_id"].startswith("SUP_")
        assert order["estimated_cost_usd"] > 0


def test_v1_optimization_product_filter():
    payload = {
        "horizon": 5,
        "product_ids": ["PROD_001"],
    }
    res = client.post("/api/v1/optimization/recommendations", json=payload)
    assert res.status_code == 200
    data = res.json()
    for order in data["recommended_orders"]:
        assert order["product_id"] == "PROD_001"


def test_v1_optimization_invalid_product():
    payload = {
        "horizon": 5,
        "product_ids": ["NON_EXISTENT_SKU"],
    }
    res = client.post("/api/v1/optimization/recommendations", json=payload)
    assert res.status_code == 422
    assert "does not exist" in res.json()["detail"]


def test_v1_optimization_invalid_horizon():
    payload = {"horizon": 50}
    res = client.post("/api/v1/optimization/recommendations", json=payload)
    assert res.status_code == 422


def test_v1_optimization_invalid_forecast_method():
    payload = {"forecast_method": "transformer_gpt"}
    res = client.post("/api/v1/optimization/recommendations", json=payload)
    assert res.status_code == 422


# -------------------------------------------------------------------------
# 7. Simulation Run Endpoint Tests
# -------------------------------------------------------------------------

def test_v1_simulation_run_small():
    payload = {
        "policy": "min_max",
        "episode_days": 10,
        "random_seed": 42,
        "n_warehouses": 1,
        "n_products": 3,
        "n_suppliers": 2,
    }
    res = client.post("/api/v1/simulation/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"].startswith("sim_")
    assert "Min-Max" in data["policy_name"]
    assert data["episode_days"] == 10
    assert data["total_steps"] == 10
    assert "fill_rate_percent" in data["kpis"]
    assert "total_cost" in data["kpis"]
    assert "procurement_cost" in data["cost_breakdown"]
    assert len(data["step_history_sample"]) > 0


def test_v1_simulation_run_invalid_policy():
    payload = {
        "policy": "magic_policy",
        "episode_days": 10,
    }
    res = client.post("/api/v1/simulation/run", json=payload)
    assert res.status_code == 422


def test_v1_simulation_run_invalid_duration():
    payload = {
        "policy": "min_max",
        "episode_days": 500,  # exceeds max 180
    }
    res = client.post("/api/v1/simulation/run", json=payload)
    assert res.status_code == 422


# -------------------------------------------------------------------------
# 8. CORS and 404 Routing
# -------------------------------------------------------------------------

def test_v1_404_routing():
    res = client.get("/api/v1/non_existent_resource")
    assert res.status_code == 404


def test_cors_options_preflight():
    res = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"


# -------------------------------------------------------------------------
# 9. Audit Tests: Forecast product_id=ALL & Benchmark Metrics
# -------------------------------------------------------------------------

def test_v1_forecasts_all_products_aggregate():
    res = client.get("/api/v1/forecasts?product_id=ALL&model=Ridge&horizon=14")
    assert res.status_code == 200
    data = res.json()
    assert data["product_id"] == "ALL"
    assert data["model_name"] == "Ridge"
    assert "Network-Wide" in data["evaluation_scope"]
    assert data["benchmark_metrics"] is not None
    assert abs(data["benchmark_metrics"]["wape_percent"] - 20.62) < 0.1
    assert abs(data["benchmark_metrics"]["mae"] - 4.21) < 0.1
    assert abs(data["benchmark_metrics"]["rmse"] - 5.78) < 0.1
    assert abs(data["benchmark_metrics"]["bias_percent"] - (-5.14)) < 0.1


# -------------------------------------------------------------------------
# 10. Audit Tests: Health Endpoint Degradation
# -------------------------------------------------------------------------

def test_v1_health_degraded_when_cbc_missing(monkeypatch):
    monkeypatch.setattr("optichain.api.routes.health.check_cbc_solver", lambda: False)
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "degraded"
    assert data["cbc_solver_available"] is False
    assert data["benchmark_artifact_present"] is True


def test_v1_health_degraded_when_benchmark_missing(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr("optichain.api.routes.health.get_optimization_benchmark_path", lambda: Path("/non/existent/path.json"))
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "degraded"
    assert data["benchmark_artifact_present"] is False


def test_v1_health_unhealthy_when_both_missing(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr("optichain.api.routes.health.check_cbc_solver", lambda: False)
    monkeypatch.setattr("optichain.api.routes.health.get_optimization_benchmark_path", lambda: Path("/non/existent/path.json"))
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "unhealthy"
    assert data["cbc_solver_available"] is False
    assert data["benchmark_artifact_present"] is False


# -------------------------------------------------------------------------
# 11. Audit Tests: 503 Error on Missing/Malformed Artifacts
# -------------------------------------------------------------------------

def test_v1_dashboard_summary_missing_artifact_503(monkeypatch):
    def fake_loader():
        raise FileNotFoundError("Simulated missing artifact")
    monkeypatch.setattr("optichain.api.routes.dashboard.load_optimization_benchmark", fake_loader)
    res = client.get("/api/v1/dashboard/summary")
    assert res.status_code == 503
    assert "Benchmark artifact unavailable" in res.json()["detail"]


def test_v1_benchmarks_missing_artifact_503(monkeypatch):
    def fake_loader():
        raise FileNotFoundError("Simulated missing artifact")
    monkeypatch.setattr("optichain.api.routes.benchmarks.load_optimization_benchmark", fake_loader)
    res = client.get("/api/v1/benchmarks")
    assert res.status_code == 503
    assert "Benchmark artifact unavailable" in res.json()["detail"]


# -------------------------------------------------------------------------
# 12. Audit Tests: Deterministic Solver Status Mappings
# -------------------------------------------------------------------------

from optichain.optimization.milp_planner import OptimizationResult

def test_v1_optimization_deterministic_status_proven_optimal(monkeypatch):
    mock_res = OptimizationResult(
        status="Optimal",
        is_optimal=True,
        objective_value=12345.0,
        planned_orders={},
        full_schedule={0: {("WH_001", "PROD_001", "SUP_001"): 50}},
        projected_inventory={},
        projected_unmet={},
        solve_time_sec=0.25,
        is_proven_optimal=True,
        is_time_limited=False,
        termination_reason="ProvenOptimal",
        has_feasible_solution=True,
        is_feasible_incumbent=False,
        is_infeasible=False,
    )
    monkeypatch.setattr("optichain.api.routes.optimization.MILPReplenishmentPlanner.solve", lambda *args, **kwargs: mock_res)
    res = client.post("/api/v1/optimization/recommendations", json={"horizon": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["solver_status"] == "Optimal"
    assert data["termination_reason"] == "ProvenOptimal"
    assert data["is_proven_optimal"] is True
    assert data["is_feasible"] is True
    assert len(data["recommended_orders"]) == 1


def test_v1_optimization_deterministic_status_feasible_incumbent_unproven(monkeypatch):
    mock_res = OptimizationResult(
        status="IntegerFeasible",
        is_optimal=False,
        objective_value=15000.0,
        planned_orders={},
        full_schedule={0: {("WH_001", "PROD_001", "SUP_001"): 40}},
        projected_inventory={},
        projected_unmet={},
        solve_time_sec=5.01,
        is_proven_optimal=False,
        is_time_limited=True,
        termination_reason="FeasibleIncumbentUnproven",
        has_feasible_solution=True,
        is_feasible_incumbent=True,
        is_infeasible=False,
    )
    monkeypatch.setattr("optichain.api.routes.optimization.MILPReplenishmentPlanner.solve", lambda *args, **kwargs: mock_res)
    res = client.post("/api/v1/optimization/recommendations", json={"horizon": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["solver_status"] == "IntegerFeasible"
    assert data["termination_reason"] == "FeasibleIncumbentUnproven"
    assert data["is_proven_optimal"] is False
    assert data["is_feasible"] is True


def test_v1_optimization_deterministic_status_infeasible(monkeypatch):
    mock_res = OptimizationResult(
        status="Infeasible",
        is_optimal=False,
        objective_value=float("inf"),
        planned_orders={},
        full_schedule={},
        projected_inventory={},
        projected_unmet={},
        solve_time_sec=0.10,
        is_proven_optimal=False,
        is_time_limited=False,
        termination_reason="Infeasible",
        has_feasible_solution=False,
        is_feasible_incumbent=False,
        is_infeasible=True,
    )
    monkeypatch.setattr("optichain.api.routes.optimization.MILPReplenishmentPlanner.solve", lambda *args, **kwargs: mock_res)
    res = client.post("/api/v1/optimization/recommendations", json={"horizon": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["solver_status"] == "Infeasible"
    assert data["termination_reason"] == "Infeasible"
    assert data["is_proven_optimal"] is False
    assert data["is_feasible"] is False
    assert len(data["recommended_orders"]) == 0


def test_v1_optimization_deterministic_status_unbounded(monkeypatch):
    mock_res = OptimizationResult(
        status="Unbounded",
        is_optimal=False,
        objective_value=float("-inf"),
        planned_orders={},
        full_schedule={},
        projected_inventory={},
        projected_unmet={},
        solve_time_sec=0.05,
        is_proven_optimal=False,
        is_time_limited=False,
        termination_reason="Unbounded",
        has_feasible_solution=False,
        is_feasible_incumbent=False,
        is_infeasible=False,
    )
    monkeypatch.setattr("optichain.api.routes.optimization.MILPReplenishmentPlanner.solve", lambda *args, **kwargs: mock_res)
    res = client.post("/api/v1/optimization/recommendations", json={"horizon": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["solver_status"] == "Unbounded"
    assert data["termination_reason"] == "Unbounded"
    assert data["is_proven_optimal"] is False
    assert data["is_feasible"] is False


def test_v1_optimization_deterministic_status_no_solution_found(monkeypatch):
    mock_res = OptimizationResult(
        status="Not Solved",
        is_optimal=False,
        objective_value=float("inf"),
        planned_orders={},
        full_schedule={},
        projected_inventory={},
        projected_unmet={},
        solve_time_sec=5.0,
        is_proven_optimal=False,
        is_time_limited=True,
        termination_reason="NoSolutionFound",
        has_feasible_solution=False,
        is_feasible_incumbent=False,
        is_infeasible=False,
    )
    monkeypatch.setattr("optichain.api.routes.optimization.MILPReplenishmentPlanner.solve", lambda *args, **kwargs: mock_res)
    res = client.post("/api/v1/optimization/recommendations", json={"horizon": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["solver_status"] == "Not Solved"
    assert data["termination_reason"] == "NoSolutionFound"
    assert data["is_proven_optimal"] is False
    assert data["is_feasible"] is False

