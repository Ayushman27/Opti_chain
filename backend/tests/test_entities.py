"""
Unit tests for OptiChain AI core domain entities and data structures.
"""

import pytest
from pydantic import ValidationError

from optichain.core.entities import (
    Product,
    Warehouse,
    Supplier,
    Order,
    OrderStatus,
    Disruption,
    DisruptionType,
    SimulationConfig,
)


def test_product_creation_and_margin():
    prod = Product(
        id="PROD_001",
        name="Wireless Earbuds",
        category="Electronics",
        unit_cost=50.0,
        selling_price=80.0,
        holding_cost_rate=0.25,
        stockout_penalty=75.0,
        weight_kg=0.3,
        volume_m3=0.001,
    )
    assert prod.id == "PROD_001"
    assert prod.unit_margin == 30.0
    assert prod.category == "Electronics"


def test_product_validation_failure_on_negative_cost():
    with pytest.raises((ValueError, ValidationError)):
        Product(
            id="PROD_NEG",
            name="Invalid",
            unit_cost=-10.0,
            selling_price=20.0,
            holding_cost_rate=0.1,
            stockout_penalty=10.0,
        )


def test_warehouse_creation():
    wh = Warehouse(
        id="WH_01",
        name="Central DC",
        location="Chicago",
        storage_capacity=50000.0,
        daily_fixed_cost=600.0,
    )
    assert wh.storage_capacity == 50000.0
    assert wh.daily_fixed_cost == 600.0


def test_supplier_creation_and_catalog():
    sup = Supplier(
        id="SUP_01",
        name="Apex Logistics",
        base_lead_time_days=3,
        lead_time_std=0.5,
        reliability=0.98,
        fixed_order_cost=120.0,
        catalog={"PROD_001": 48.0},
        min_order_quantity={"PROD_001": 10},
        daily_capacity={"PROD_001": 500},
    )
    assert sup.catalog["PROD_001"] == 48.0
    assert sup.reliability == 0.98


def test_order_lifecycle():
    order = Order(
        id="ORD_001",
        step_created=0,
        supplier_id="SUP_01",
        warehouse_id="WH_01",
        product_id="PROD_001",
        quantity=100,
        unit_cost=45.0,
        fixed_order_cost=50.0,
        transport_cost=150.0,
        expected_arrival_step=3,
    )
    assert order.status == OrderStatus.IN_TRANSIT
    assert order.total_cost == (100 * 45.0) + 50.0 + 150.0


def test_disruption_activity_window():
    disr = Disruption(
        id="DISR_01",
        name="Port Strike",
        type=DisruptionType.PORT_CONGESTION,
        start_step=10,
        duration_steps=5,
        impact_multiplier=2.0,
    )
    assert not disr.is_active(9)
    assert disr.is_active(10)
    assert disr.is_active(14)
    assert not disr.is_active(15)


def test_simulation_config_defaults():
    cfg = SimulationConfig()
    assert cfg.n_warehouses == 1
    assert cfg.n_products == 5
    assert cfg.n_suppliers == 3
    assert cfg.episode_days == 90
