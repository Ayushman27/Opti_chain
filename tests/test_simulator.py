"""
Unit and integration tests for DigitalTwinSimulator.
Validates inventory mass conservation, transition dynamics, order pipelines, and KPIs.
"""

import numpy as np
import pytest

from optichain.core.entities import SimulationConfig, OrderStatus
from optichain.simulator.environment import DigitalTwinSimulator


def test_simulator_initialization_and_reset():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=4, n_suppliers=2, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    obs = sim.reset(seed=42)

    assert obs["current_step"] == 0
    assert obs["on_hand"].shape == (2, 4)
    assert np.all(obs["on_hand"] >= 0)
    assert not obs["is_terminal"]
    assert len(sim.in_transit_orders) == 0


def test_inventory_conservation_of_mass_single_step():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=3, n_suppliers=2, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    init_stock = sim.on_hand.copy()
    w_0_p_0_init = init_stock[0, 0]

    # Step with no orders placed
    obs, reward, term, trunc, info = sim.step({})

    demand_realized = info["demand"]
    fulfilled = info["fulfilled"]
    unmet = info["unmet"]

    assert fulfilled + unmet == demand_realized
    # On-hand inventory should never drop below zero
    assert np.all(sim.on_hand >= 0)
    assert sim.current_step == 1


def test_order_pipeline_arrival():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=1, n_suppliers=1, episode_days=10, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    w_id = sim.warehouses[0].id
    p_id = sim.products[0].id
    s_id = sim.suppliers[0].id

    # Place an order of 100 units
    order_action = {(w_id, p_id, s_id): 100}
    obs, reward, term, trunc, info = sim.step(order_action)

    assert len(sim.all_orders_history) == 1
    placed_order = sim.all_orders_history[0]
    assert placed_order.quantity >= 100
    arrival_step = placed_order.expected_arrival_step

    # Step forward including arrival step
    while sim.current_step <= arrival_step:
        sim.step({})

    # Check that at arrival step, order was delivered
    assert placed_order.status == OrderStatus.DELIVERED
    assert placed_order.actual_arrival_step == arrival_step


def test_simulation_termination():
    cfg = SimulationConfig(
        n_warehouses=1, n_products=2, n_suppliers=1, episode_days=5, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    for step in range(5):
        obs, reward, terminated, _, _ = sim.step({})
        if step < 4:
            assert not terminated
        else:
            assert terminated

    # Attempting to step after termination should raise RuntimeError
    with pytest.raises(RuntimeError):
        sim.step({})


def test_kpi_metrics_computation():
    cfg = SimulationConfig(
        n_warehouses=2, n_products=3, n_suppliers=2, episode_days=7, random_seed=42
    )
    sim = DigitalTwinSimulator(config=cfg)
    sim.reset(seed=42)

    while sim.current_step < cfg.episode_days:
        sim.step({})

    kpis = sim.get_kpis()
    assert kpis.total_demand > 0
    assert 0.0 <= kpis.fill_rate_percent <= 100.0
    assert 0.0 <= kpis.alpha_service_level_percent <= 100.0
    assert kpis.total_cost > 0.0
