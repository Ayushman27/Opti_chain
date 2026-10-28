"""
OptiChain AI - Unit and Integration Tests for Operations Research MILP Optimization
Tests:
1. Action generation & horizon execution
2. Inventory conservation of mass across planning horizon
3. Supplier catalog eligibility enforcement
4. Minimum order quantity (MOQ) and daily capacity constraints
5. Fixed order setup cost binary linking
6. Infeasible scenario detection and graceful fallback
7. Zero lookahead leakage (future actual demand does not alter current decision)
8. End-to-end evaluation with complete cost breakdown & KPI metrics
"""

import numpy as np
import pytest
import pulp

from optichain.config import SimulationConfig
from optichain.data.generator import SupplyChainDataGenerator
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.optimization.milp_planner import (
    MILPReplenishmentPolicy,
    MILPReplenishmentPlanner,
    OptimizationResult,
)


def test_milp_planner_action_generation():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    obs = sim.reset(seed=42)

    planner = MILPReplenishmentPolicy(horizon=3, time_limit_sec=3)
    planner.fit(sim)

    actions = planner.get_actions(obs, sim)
    assert isinstance(actions, dict)
    for (w_id, p_id, s_id), qty in actions.items():
        assert w_id == sim.warehouses[0].id
        assert qty >= 0


def test_milp_planner_episode_execution():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=4, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    planner = MILPReplenishmentPolicy(horizon=3, time_limit_sec=3)
    evaluator = PolicyEvaluator(cfg)

    result = evaluator.run_episode(planner, sim, seed=42)
    kpis = result["kpis"]

    assert kpis.total_demand > 0
    assert len(result["step_history"]) == 4
    assert kpis.fill_rate_percent >= 0.0


def test_milp_inventory_conservation_mathematics():
    """
    Directly tests that the MILP solution satisfies inventory balance for every epoch tau:
    I_{tau} = I_{tau-1} + Arrivals - ForecastDemand + UnmetDemand.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    H = 4
    W = len(sim.warehouses)
    P = len(sim.products)
    current_on_hand = np.array([[10.0, 20.0]])
    scheduled_arrivals = np.zeros((H, W, P), dtype=float)
    forecast_demands = np.array([
        [[15.0, 25.0]],
        [[12.0, 18.0]],
        [[20.0, 30.0]],
        [[10.0, 15.0]],
    ])

    solver = MILPReplenishmentPlanner(time_limit_sec=3.0)
    res = solver.solve(
        current_step=0,
        current_on_hand=current_on_hand,
        scheduled_arrivals=scheduled_arrivals,
        forecast_demands=forecast_demands,
        warehouses=sim.warehouses,
        products=sim.products,
        suppliers=sim.suppliers,
        horizon=H,
    )

    assert res.is_optimal
    assert res.objective_value < float("inf")

    # Verify balance for product 0 and 1
    for w in range(W):
        for p in range(P):
            for tau in range(H):
                I_val = res.projected_inventory[(w, p, tau)]
                U_val = res.projected_unmet[(w, p, tau)]
                prev_I = current_on_hand[w, p] if tau == 0 else res.projected_inventory[(w, p, tau - 1)]
                dem = forecast_demands[tau, w, p]

                # Incoming from horizon orders
                arr_h = 0.0
                for s_i, s in enumerate(sim.suppliers):
                    order_epoch = tau - s.base_lead_time_days
                    if order_epoch >= 0:
                        key = (sim.warehouses[w].id, sim.products[p].id, s.id)
                        arr_h += res.full_schedule.get(order_epoch, {}).get(key, 0)

                total_incoming = scheduled_arrivals[tau, w, p] + arr_h
                # Balance: I_tau == prev_I + Incoming - Demand + Unmet
                expected_I = prev_I + total_incoming - dem + U_val
                assert I_val == pytest.approx(expected_I, abs=1e-3)
                assert I_val >= -1e-4
                assert U_val >= -1e-4


def test_milp_supplier_catalog_eligibility_enforcement():
    """
    Verifies that no orders are generated for products missing from a supplier's catalog.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=3, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    # Artificially remove product 0 from supplier 0's catalog
    p0_id = sim.products[0].id
    s0 = sim.suppliers[0]
    if p0_id in s0.catalog:
        del s0.catalog[p0_id]

    solver = MILPReplenishmentPlanner(time_limit_sec=2.0)
    res = solver.solve(
        current_step=0,
        current_on_hand=np.zeros((1, 3)),
        scheduled_arrivals=np.zeros((3, 1, 3)),
        forecast_demands=np.full((3, 1, 3), 20.0),
        warehouses=sim.warehouses,
        products=sim.products,
        suppliers=sim.suppliers,
        horizon=3,
    )

    # Check across entire schedule that (w0, p0, s0) is NEVER ordered
    for tau, orders in res.full_schedule.items():
        assert (sim.warehouses[0].id, p0_id, s0.id) not in orders


def test_milp_moq_and_daily_capacity_constraints():
    """
    Verifies that all non-zero orders generated by the MILP satisfy MOQ and Daily Capacity.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    # Set explicit MOQ and capacity on supplier 0, product 0
    s0 = sim.suppliers[0]
    p0 = sim.products[0]
    s0.catalog[p0.id] = 10.0
    s0.min_order_quantity[p0.id] = 25
    s0.daily_capacity[p0.id] = 100

    solver = MILPReplenishmentPlanner(time_limit_sec=2.0)
    res = solver.solve(
        current_step=0,
        current_on_hand=np.zeros((1, 2)),
        scheduled_arrivals=np.zeros((4, 1, 2)),
        forecast_demands=np.full((4, 1, 2), 30.0),
        warehouses=sim.warehouses,
        products=sim.products,
        suppliers=sim.suppliers,
        horizon=4,
        enable_moq=True,
    )

    assert res.is_optimal
    for tau, orders in res.full_schedule.items():
        key = (sim.warehouses[0].id, p0.id, s0.id)
        if key in orders:
            qty = orders[key]
            assert qty >= s0.min_order_quantity[p0.id], f"Order {qty} below MOQ {s0.min_order_quantity[p0.id]}"
            assert qty <= s0.daily_capacity[p0.id], f"Order {qty} exceeds Capacity {s0.daily_capacity[p0.id]}"


def test_milp_infeasible_scenario_fallback():
    """
    Tests graceful degradation when constraints render the problem mathematically infeasible.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    # Make warehouse storage capacity 0 with initial inventory > 0
    sim.warehouses[0].storage_capacity = -10.0  # Impossible storage limit

    policy = MILPReplenishmentPolicy(horizon=3, time_limit_sec=1.0)
    policy.fit(sim)

    obs = sim.get_observation()
    # Must not raise unhandled exception; must return a valid actions dict (e.g. empty)
    actions = policy.get_actions(obs, sim)
    assert isinstance(actions, dict)
    assert len(actions) == 0


def test_milp_zero_future_demand_leakage():
    """
    Verifies that the MILP policy generates actions based on the forecast,
    and modifying actual future ground-truth demand in sim.demand_matrix does not alter the decision.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    obs = sim.reset(seed=42)

    policy = MILPReplenishmentPolicy(horizon=4, time_limit_sec=2.0)
    policy.fit(sim)

    # Action on original simulator
    actions_orig = policy.get_actions(obs, sim)

    # Mutate future demand at steps t+2, t+3 in sim.demand_matrix to huge values
    sim.demand_matrix[2:, :, :] = 99999.0

    # Policy action should be identical because it does NOT read future sim.demand_matrix
    actions_mutated = policy.get_actions(obs, sim)

    assert actions_orig == actions_mutated


def test_milp_end_to_end_evaluation_metrics():
    """
    Verifies complete end-to-end episode execution, computing all required KPIs and cost breakdowns.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=3, n_suppliers=2, episode_days=15, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = MILPReplenishmentPolicy(name="Test MILP", horizon=5, time_limit_sec=2.0)
    evaluator = PolicyEvaluator(cfg)

    result = evaluator.run_episode(policy, sim, seed=42)
    kpis = result["kpis"]
    cost_breakdown = result["cost_breakdown"]

    assert kpis.total_demand > 0
    assert kpis.total_cost > 0
    assert 0.0 <= kpis.fill_rate_percent <= 100.0
    assert 0.0 <= kpis.alpha_service_level_percent <= 100.0
    assert kpis.total_unmet_demand >= 0
    assert kpis.average_on_hand_inventory >= 0

    # Verify complete cost breakdown fields are present and non-negative
    for cost_key in [
        "holding_cost",
        "stockout_penalty",
        "procurement_cost",
        "fixed_order_cost",
        "transport_cost",
        "operating_cost",
    ]:
        assert cost_key in cost_breakdown
        assert cost_breakdown[cost_key] >= 0.0


def test_milp_lead_time_strict_pipeline_arrival():
    """
    Regression Test: Confirm orders cannot arrive before their modeled lead time.
    When supplier has lead time L=2 days, an order placed at tau=0 cannot arrive at tau=0 or tau=1.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=1, n_suppliers=1, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    s0 = sim.suppliers[0]
    p0 = sim.products[0]
    s0.base_lead_time_days = 2

    # Start with 0 inventory, 0 pipeline, positive demand
    H = 4
    current_on_hand = np.zeros((1, 1))
    scheduled_arrivals = np.zeros((H, 1, 1))
    forecast_demands = np.array([[[10.0]], [[10.0]], [[10.0]], [[10.0]]])

    solver = MILPReplenishmentPlanner(time_limit_sec=2.0)
    res = solver.solve(
        current_step=0,
        current_on_hand=current_on_hand,
        scheduled_arrivals=scheduled_arrivals,
        forecast_demands=forecast_demands,
        warehouses=sim.warehouses,
        products=sim.products,
        suppliers=sim.suppliers,
        horizon=H,
    )

    assert res.is_optimal
    # Because lead time is 2 days and initial stock is 0,
    # at tau=0: inventory cannot be replenished, so unmet must be 10.0
    assert res.projected_unmet[(0, 0, 0)] == pytest.approx(10.0, abs=1e-3)
    # at tau=1: inventory cannot be replenished from order placed at tau=0, so unmet must be 10.0
    assert res.projected_unmet[(0, 0, 1)] == pytest.approx(10.0, abs=1e-3)
    # Orders placed at tau=0 can only arrive at tau=2:
    key = (sim.warehouses[0].id, p0.id, s0.id)
    order_at_0 = res.full_schedule.get(0, {}).get(key, 0)
    assert order_at_0 >= 0


def test_milp_solver_status_tracking():
    """
    Regression Test: Check that every solver invocation records valid status and diagnostics.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = MILPReplenishmentPolicy(horizon=4, time_limit_sec=2.0)
    evaluator = PolicyEvaluator(cfg)
    evaluator.run_episode(policy, sim, seed=42)

    assert len(policy.solve_history) == 5
    for solve_res in policy.solve_history:
        assert isinstance(solve_res, OptimizationResult)
        assert solve_res.status in ["Optimal", "Infeasible", "Not Solved", "Undefined"]
        assert solve_res.is_optimal is True
        assert solve_res.solve_time_sec >= 0.0
        assert solve_res.objective_value < float("inf")


def test_milp_cost_and_profit_exact_reconciliation():
    """
    Regression Test: Verify exact mathematical reconciliation of total cost vs cost categories,
    and net profit vs revenue minus total cost.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = MILPReplenishmentPolicy(horizon=4, time_limit_sec=2.0)
    evaluator = PolicyEvaluator(cfg)
    ep_res = evaluator.run_episode(policy, sim, seed=42)

    kpis = ep_res["kpis"]
    b = ep_res["cost_breakdown"]

    summed_costs = (
        b["holding_cost"]
        + b["stockout_penalty"]
        + b["procurement_cost"]
        + b["fixed_order_cost"]
        + b["transport_cost"]
        + b["operating_cost"]
    )
    assert kpis.total_cost == pytest.approx(summed_costs, abs=1e-2)
    assert kpis.net_profit == pytest.approx(b["revenue"] - kpis.total_cost, abs=1e-2)


def test_milp_forecasting_model_dependency():
    """
    Regression Test: Proves that the optimizer's forecasts depend directly on the intended forecasting model.
    When forecast_method='seasonal_naive' or 'ridge', predictions invoke the underlying trained model.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    obs = sim.reset(seed=42)

    # Initialize policy with seasonal naive forecasting
    policy_naive = MILPReplenishmentPolicy(
        horizon=4, forecast_method="seasonal_naive", safety_buffer_factor=0.0
    )
    policy_naive.fit(sim)

    # Populate dummy history
    policy_naive.historical_demands[(0, 0)] = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0]

    # Monkey-patch seasonal naive model to return an unmistakable test marker value (e.g. 77.0)
    original_predict = policy_naive.forecast_engine.seasonal_naive_model.predict
    try:
        policy_naive.forecast_engine.seasonal_naive_model.predict = lambda df: np.full(len(df), 77.0)
        forecasts = policy_naive._generate_forecast_window(obs, sim, H=2)
        # Verify that the generated forecast window directly reflects the monkey-patched model's prediction
        assert forecasts[0, 0, 0] == 77.0
        assert forecasts[1, 0, 0] == 77.0
    finally:
        policy_naive.forecast_engine.seasonal_naive_model.predict = original_predict


@pytest.mark.parametrize("method", ["heuristic_lag7", "seasonal_naive", "ridge", "lgbm"])
def test_all_milp_forecast_methods_execution_and_leakage(method):
    """
    Regression Test: Verifies all supported forecast methods execute cleanly end-to-end
    and that feature generation and training strictly avoid future ground-truth demand leakage.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = MILPReplenishmentPolicy(
        horizon=3, forecast_method=method, time_limit_sec=2.0
    )
    evaluator = PolicyEvaluator(cfg)
    res = evaluator.run_episode(policy, sim, seed=42)
    assert res["kpis"].total_cost > 0
    assert len(policy.solve_history) == 5

    # Verify zero future demand leakage for this method:
    # Corrupt future demand in sim.demand_matrix and ensure actions at step 0 remain identical
    sim_base = DigitalTwinSimulator(config=cfg)
    obs_base = sim_base.reset(seed=42)
    pol_test = MILPReplenishmentPolicy(horizon=3, forecast_method=method, time_limit_sec=2.0)
    pol_test.fit(sim_base)
    acts_base = pol_test.get_actions(obs_base, sim_base)

    sim_corrupted = DigitalTwinSimulator(config=cfg)
    obs_corrupted = sim_corrupted.reset(seed=42)
    # Drastically mutate future demand at epochs >= 1
    sim_corrupted.demand_matrix[1:, :, :] = 99999.0
    pol_corrupted = MILPReplenishmentPolicy(horizon=3, forecast_method=method, time_limit_sec=2.0)
    pol_corrupted.fit(sim_corrupted)
    acts_corrupted = pol_corrupted.get_actions(obs_corrupted, sim_corrupted)

    assert acts_base == acts_corrupted


@pytest.mark.parametrize("method,model_attr", [
    ("seasonal_naive", "seasonal_naive_model"),
    ("ridge", "ridge_model"),
    ("lgbm", "lgbm_model"),
])
def test_milp_model_driven_prediction_call_verification(method, model_attr):
    """
    Regression Test: Verifies that model-driven forecast methods really invoke .predict()
    on their intended underlying trained models during decision action generation.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=4, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    policy = MILPReplenishmentPolicy(
        horizon=3, forecast_method=method, time_limit_sec=2.0
    )
    policy.fit(sim)
    obs = sim.reset(seed=42)

    target_model = getattr(policy.forecast_engine, model_attr)
    call_count = [0]
    orig_predict = target_model.predict

    def tracking_predict(*args, **kwargs):
        call_count[0] += 1
        return orig_predict(*args, **kwargs)

    target_model.predict = tracking_predict
    try:
        policy.get_actions(obs, sim)
        assert call_count[0] > 0, f"Method '{method}' did not invoke {model_attr}.predict()!"
    finally:
        target_model.predict = orig_predict


def test_milp_solver_termination_classification_distinction():
    """
    Regression Test: Proves solver termination classification accurately distinguishes:
    1. Proven optimal solutions
    2. Feasible incumbents with optimality unproven (e.g. time-limited)
    3. Infeasible problems
    4. Time-limited solves with no solution found
    WITHOUT relying on elapsed-time thresholds.
    """
    class MockProblem:
        def __init__(self, status: int, sol_status: int):
            self.status = status
            self.sol_status = sol_status

    # 1. Proven optimal
    prob_optimal = MockProblem(status=pulp.constants.LpStatusOptimal, sol_status=pulp.constants.LpSolutionOptimal)
    diag_opt = MILPReplenishmentPlanner.classify_solver_termination(prob_optimal, has_vars=True)
    assert diag_opt["is_proven_optimal"] is True
    assert diag_opt["is_feasible_incumbent"] is False
    assert diag_opt["has_feasible_solution"] is True
    assert diag_opt["is_time_limited"] is False
    assert diag_opt["termination_reason"] == "ProvenOptimal"
    assert diag_opt["raw_status"] == pulp.constants.LpStatusOptimal
    assert diag_opt["raw_sol_status"] == pulp.constants.LpSolutionOptimal

    # 2. Feasible incumbent with optimality unproven (CBC stopped on time with integer incumbent)
    prob_incumbent = MockProblem(status=pulp.constants.LpStatusOptimal, sol_status=pulp.constants.LpSolutionIntegerFeasible)
    diag_inc = MILPReplenishmentPlanner.classify_solver_termination(prob_incumbent, has_vars=True)
    assert diag_inc["is_proven_optimal"] is False
    assert diag_inc["is_feasible_incumbent"] is True
    assert diag_inc["has_feasible_solution"] is True
    assert diag_inc["is_time_limited"] is True
    assert diag_inc["termination_reason"] == "FeasibleIncumbentUnproven"

    # 3. Infeasible problem
    prob_infeasible = MockProblem(status=pulp.constants.LpStatusInfeasible, sol_status=pulp.constants.LpSolutionInfeasible)
    diag_inf = MILPReplenishmentPlanner.classify_solver_termination(prob_infeasible, has_vars=False)
    assert diag_inf["is_proven_optimal"] is False
    assert diag_inf["is_infeasible"] is True
    assert diag_inf["has_feasible_solution"] is False
    assert diag_inf["termination_reason"] == "Infeasible"

    # 4. Stopped on time with NO feasible solution
    prob_nosol = MockProblem(status=pulp.constants.LpStatusNotSolved, sol_status=pulp.constants.LpSolutionNoSolutionFound)
    diag_nosol = MILPReplenishmentPlanner.classify_solver_termination(prob_nosol, has_vars=False)
    assert diag_nosol["has_feasible_solution"] is False
    assert diag_nosol["is_time_limited"] is True
    assert diag_nosol["termination_reason"] == "NoSolutionFound"


@pytest.mark.parametrize("method", ["ridge", "lgbm"])
def test_forecast_training_data_leakage_prevention_on_future_demand_mutation(method):
    """
    Regression Test: Proves that ML forecasting models (Ridge, LightGBM) are fitted exclusively
    on pre-simulation historical data (cutoff at Day 0) and NEVER absorb future evaluation demand.
    Mutating future simulation demand before model fitting has zero impact on training data,
    learned model parameters, and step 0 policy actions.
    """
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=2, episode_days=10, random_seed=42
    )

    # 1. Base simulation environment
    sim_base = DigitalTwinSimulator(config=cfg)
    obs_base = sim_base.reset(seed=42)

    # 2. Corrupted future demand environment (mutating future demand BEFORE policy.fit())
    sim_mutated = DigitalTwinSimulator(config=cfg)
    obs_mutated = sim_mutated.reset(seed=42)
    # Mutate all future simulation demand across the evaluation episode
    sim_mutated.demand_matrix[:, :, :] = 99999.0

    # Ensure pre-simulation historical datasets are intact and identical (cutoff at Day 0)
    np.testing.assert_array_equal(sim_base.historical_demand_matrix, sim_mutated.historical_demand_matrix)

    # Fit policies on both environments
    pol_base = MILPReplenishmentPolicy(horizon=4, forecast_method=method, time_limit_sec=2.0)
    pol_mutated = MILPReplenishmentPolicy(horizon=4, forecast_method=method, time_limit_sec=2.0)

    pol_base.fit(sim_base)
    pol_mutated.fit(sim_mutated)

    # Verify model parameters are completely identical
    if method == "ridge":
        ridge_base = pol_base.forecast_engine.ridge_model.pipeline.named_steps["ridge"]
        ridge_mutated = pol_mutated.forecast_engine.ridge_model.pipeline.named_steps["ridge"]
        np.testing.assert_array_almost_equal(ridge_base.coef_, ridge_mutated.coef_)
        assert ridge_base.intercept_ == pytest.approx(ridge_mutated.intercept_)
    elif method == "lgbm":
        lgbm_base = pol_base.forecast_engine.lgbm_model.model
        lgbm_mutated = pol_mutated.forecast_engine.lgbm_model.model
        np.testing.assert_array_almost_equal(
            lgbm_base.feature_importances_, lgbm_mutated.feature_importances_
        )

    # Verify generated actions at step 0 remain strictly identical
    acts_base = pol_base.get_actions(obs_base, sim_base)
    acts_mutated = pol_mutated.get_actions(obs_mutated, sim_mutated)
    assert acts_base == acts_mutated




