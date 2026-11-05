"""
OptiChain AI - Phase 2: Digital Twin Simulator & Heuristic Baselines Tests
Comprehensive unit and integration test suite covering:
- Inventory balance & conservation of mass
- Delivery scheduling & outstanding-order pipeline tracking
- Zero inventory, excess inventory, and stockouts
- Supplier catalog eligibility, MOQ, and daily capacity constraints
- Granular cost calculations & KPI edge cases
- Disruption impact & fixed-seed reproducibility
- End-to-end multi-seed benchmark execution
"""

import numpy as np
import pytest

from optichain.config import SimulationConfig, get_default_config
from optichain.core.entities import (
    Product,
    Warehouse,
    Supplier,
    Order,
    OrderStatus,
    DisruptionEvent,
    DisruptionType,
)
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy


# --------------------------------------------------------------------------
# 1. Inventory Balance & Conservation of Mass
# --------------------------------------------------------------------------

def test_inventory_balance_and_demand_conservation():
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=10, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    for step in range(10):
        # Record pre-step on-hand
        pre_on_hand = sim.on_hand.copy()

        # Place a small replenishment order every 3 days
        actions = {}
        if step % 3 == 0:
            actions[(sim.warehouses[0].id, sim.products[0].id, sim.suppliers[0].id)] = 25

        obs, reward, term, trunc, info = sim.step(actions)

        # 1. Demand conservation: Fulfilled + Unmet == Realized Customer Demand
        demand = info["demand"]
        fulfilled = info["fulfilled"]
        unmet = info["unmet"]
        assert fulfilled + unmet == demand, f"Demand conservation violated at step {step}"

        # 2. Inventory non-negativity: On-hand stock is strictly >= 0
        assert np.all(sim.on_hand >= 0), f"Negative inventory detected at step {step}"

        # 3. Mass balance check for every product
        for p in range(2):
            dem_p = sim.history_demand[step, 0, p]
            ful_p = sim.history_fulfilled[step, 0, p]
            unm_p = sim.history_unmet[step, 0, p]
            arr_p = sim.history_arrivals[step, 0, p]
            post_on_hand_p = sim.history_on_hand[step, 0, p]

            assert ful_p + unm_p == dem_p
            expected_post_stock = max(0.0, pre_on_hand[0, p] + arr_p - dem_p)
            assert post_on_hand_p == pytest.approx(expected_post_stock, abs=1e-4)


# --------------------------------------------------------------------------
# 2. Delivery Scheduling & Outstanding Order Tracking
# --------------------------------------------------------------------------

def test_delivery_scheduling_and_lead_time_arrival():
    cfg = SimulationConfig(n_warehouses=1, n_products=1, n_suppliers=1, episode_days=15, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    w_id = sim.warehouses[0].id
    p_id = sim.products[0].id
    s_id = sim.suppliers[0].id

    # Place an order at step 0
    obs, reward, term, trunc, info = sim.step({(w_id, p_id, s_id): 50})
    assert len(sim.all_orders_history) == 1
    order = sim.all_orders_history[0]
    expected_arr = order.expected_arrival_step
    assert expected_arr >= 1, "Arrival step must be in the future (L >= 1)"

    # Prior to arrival step, the order must remain IN_TRANSIT
    while sim.current_step < expected_arr:
        assert order.status == OrderStatus.IN_TRANSIT
        assert any(o.id == order.id for o in sim.in_transit_orders)
        sim.step({})

    # Execute step at arrival day
    pre_arrival_stock = sim.on_hand[0, 0]
    obs, reward, term, trunc, info = sim.step({})

    # Order must now be marked DELIVERED and removed from in_transit_orders
    assert order.status == OrderStatus.DELIVERED
    assert order.actual_arrival_step == expected_arr
    assert not any(o.id == order.id for o in sim.in_transit_orders)
    # Check that arrival units are recorded
    assert info["arrived_units"] == order.quantity


# --------------------------------------------------------------------------
# 3. Zero Inventory, Excess Inventory & Stockouts
# --------------------------------------------------------------------------

def test_zero_inventory_stockout_accounting():
    cfg = SimulationConfig(n_warehouses=1, n_products=1, n_suppliers=1, episode_days=3, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)
    # Force on-hand inventory to 0
    sim.on_hand.fill(0.0)

    obs, reward, term, trunc, info = sim.step({})
    demand = info["demand"]

    if demand > 0:
        assert info["fulfilled"] == 0
        assert info["unmet"] == demand
        assert info["fill_rate"] == 0.0
        assert sim.on_hand[0, 0] == 0.0, "Zero stock must not become negative"
        assert sim.cost_engine.history[-1].stockout_penalty > 0.0
        assert sim.cost_engine.history[-1].holding_cost == 0.0


def test_excess_inventory_holding_cost():
    cfg = SimulationConfig(n_warehouses=1, n_products=1, n_suppliers=1, episode_days=3, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)
    # Inject massive excess inventory
    sim.on_hand.fill(50000.0)

    obs, reward, term, trunc, info = sim.step({})
    demand = info["demand"]

    assert info["fulfilled"] == demand
    assert info["unmet"] == 0
    assert info["fill_rate"] == 100.0
    assert sim.cost_engine.history[-1].stockout_penalty == 0.0
    assert sim.cost_engine.history[-1].holding_cost > 0.0


# --------------------------------------------------------------------------
# 4. Supplier Catalog, MOQ & Daily Capacity Constraints
# --------------------------------------------------------------------------

def test_supplier_catalog_eligibility_rejection():
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=5, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    w_id = sim.warehouses[0].id
    p_valid = sim.products[0].id
    p_invalid = "PROD_UNAVAILABLE"
    s_id = sim.suppliers[0].id

    # Ensure p_invalid is NOT in supplier catalog
    assert p_invalid not in sim.suppliers[0].catalog

    # Attempt to order invalid SKU
    sim.step({(w_id, p_invalid, s_id): 100})
    assert len(sim.all_orders_history) == 0, "Ineligible catalog orders must be rejected"


def test_supplier_moq_enforcement():
    cfg = SimulationConfig(n_warehouses=1, n_products=1, n_suppliers=1, episode_days=5, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    w_id = sim.warehouses[0].id
    p_id = sim.products[0].id
    s_id = sim.suppliers[0].id

    # Set explicit MOQ of 25 units
    sim.suppliers[0].min_order_quantity[p_id] = 25

    # Request only 2 units (below MOQ)
    sim.step({(w_id, p_id, s_id): 2})
    assert len(sim.all_orders_history) == 1
    placed_order = sim.all_orders_history[0]
    assert placed_order.quantity == 25, "Order below MOQ must be elevated to supplier MOQ threshold"


def test_supplier_daily_capacity_clipping():
    cfg = SimulationConfig(n_warehouses=1, n_products=1, n_suppliers=1, episode_days=5, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    w_id = sim.warehouses[0].id
    p_id = sim.products[0].id
    s_id = sim.suppliers[0].id

    # Set supplier daily capacity to 40 units
    sim.suppliers[0].daily_capacity[p_id] = 40

    # Request 150 units (exceeds capacity)
    sim.step({(w_id, p_id, s_id): 150})
    assert len(sim.all_orders_history) == 1
    placed_order = sim.all_orders_history[0]
    assert placed_order.quantity == 40, "Order exceeding daily capacity must be clipped to capacity limit"


# --------------------------------------------------------------------------
# 5. Granular Cost Accounting & Mathematical Correctness
# --------------------------------------------------------------------------

def test_cost_accounting_exact_formula():
    cfg = SimulationConfig(n_warehouses=1, n_products=1, n_suppliers=1, episode_days=5, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    w_id = sim.warehouses[0].id
    p_id = sim.products[0].id
    s_id = sim.suppliers[0].id

    obs, reward, term, trunc, info = sim.step({(w_id, p_id, s_id): 30})
    cost_summary = sim.cost_engine.get_summary()

    # Verify exact summation formula
    expected_total = (
        cost_summary["holding_cost"]
        + cost_summary["stockout_penalty"]
        + cost_summary["fixed_order_cost"]
        + cost_summary["procurement_cost"]
        + cost_summary["transport_cost"]
        + cost_summary["operating_cost"]
    )
    assert cost_summary["total_cost"] == pytest.approx(expected_total, abs=1e-2)
    assert cost_summary["net_profit"] == pytest.approx(
        cost_summary["revenue"] - cost_summary["total_cost"], abs=1e-2
    )


# --------------------------------------------------------------------------
# 6. Disruption Handling & Lead Time Expansion
# --------------------------------------------------------------------------

def test_disruption_lead_time_expansion():
    cfg = SimulationConfig(n_warehouses=1, n_products=1, n_suppliers=1, episode_days=20, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    s_id = sim.suppliers[0].id
    # Add disruption: days 5 to 10 with 2.5x lead time multiplier
    disr = DisruptionEvent(
        id="DISR_DELAY",
        name="Port Delay",
        type=DisruptionType.PORT_CONGESTION,
        start_step=5,
        duration_steps=6,
        impact_multiplier=2.5,
    )
    sim.disruptions = [disr]

    # Sample lead time outside disruption window (step 0)
    normal_lts = [sim._sample_lead_time(sim.suppliers[0], current_step=0) for _ in range(100)]
    # Sample lead time inside disruption window (step 7)
    disrupted_lts = [sim._sample_lead_time(sim.suppliers[0], current_step=7) for _ in range(100)]

    assert np.mean(disrupted_lts) > np.mean(normal_lts) * 1.8


# --------------------------------------------------------------------------
# 7. Fixed Seed Simulation Determinism
# --------------------------------------------------------------------------

def test_fixed_seed_simulation_determinism():
    cfg = SimulationConfig(n_warehouses=1, n_products=5, n_suppliers=3, episode_days=15, random_seed=42)

    sim1 = DigitalTwinSimulator(config=cfg)
    sim2 = DigitalTwinSimulator(config=cfg)

    policy = MinMaxPolicy()
    evaluator = PolicyEvaluator(cfg)

    res1 = evaluator.run_episode(policy, sim1, seed=42)
    res2 = evaluator.run_episode(policy, sim2, seed=42)

    assert res1["kpis"].total_demand == res2["kpis"].total_demand
    assert res1["kpis"].total_fulfilled == res2["kpis"].total_fulfilled
    assert res1["kpis"].total_cost == res2["kpis"].total_cost
    assert res1["kpis"].net_profit == res2["kpis"].net_profit


# --------------------------------------------------------------------------
# 8. End-to-End Heuristic Benchmark Execution
# --------------------------------------------------------------------------

def test_end_to_end_heuristic_benchmark():
    cfg = get_default_config()  # 1 WH, 5 Products, 3 Suppliers, 90 Days
    evaluator = PolicyEvaluator(cfg)

    policies = [
        ROPPolicy(name="ROP Order-Up-To Target"),
        MinMaxPolicy(name="Min-Max (s, S)"),
        PeriodicReviewPolicy(name="Periodic Review (R, S) 7-day", review_period_days=7),
    ]

    df = evaluator.benchmark(policies, seeds=[42, 101])

    assert len(df) == 3
    expected_cols = [
        "Policy",
        "Fill Rate (%)",
        "Service Level (%)",
        "Total Cost ($)",
        "Stockout Units",
        "Average Inventory (Units)",
        "Holding Cost ($)",
        "Stockout Cost ($)",
        "Procurement Cost ($)",
        "Net Profit ($)",
        "Inventory Turnover",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Every policy must maintain a positive fill rate and valid non-negative metrics
    for _, row in df.iterrows():
        assert 0.0 <= row["Fill Rate (%)"] <= 100.0
        assert 0.0 <= row["Service Level (%)"] <= 100.0
        assert row["Total Cost ($)"] > 0.0
        assert row["Average Inventory (Units)"] >= 0.0
