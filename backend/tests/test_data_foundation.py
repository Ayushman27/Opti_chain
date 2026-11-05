"""
OptiChain AI - Phase 1: Business Problem & Synthetic Data Foundation Tests
Validates typed dataclasses, configuration constraints, demand generation invariants,
reproducibility, and stochastic supplier lead-time constraints.
"""

import numpy as np
import pytest

from optichain.config import SimulationConfig, get_default_config, get_benchmark_config
from optichain.core.entities import (
    Product,
    Warehouse,
    Supplier,
    DisruptionEvent,
    DisruptionType,
    OrderStatus,
)
from optichain.data.generator import SupplyChainDataGenerator


# --------------------------------------------------------------------------
# 1. Typed Dataclass & Configuration Validation
# --------------------------------------------------------------------------

def test_simulation_config_phase1_defaults():
    cfg = get_default_config()
    assert cfg.n_warehouses == 1
    assert cfg.n_products == 5
    assert cfg.n_suppliers == 3
    assert cfg.episode_days == 90
    assert cfg.random_seed == 42
    assert cfg.transport_cost_per_unit == 1.5


def test_simulation_config_invalid_constraints():
    with pytest.raises(ValueError, match="n_warehouses"):
        SimulationConfig(n_warehouses=0)

    with pytest.raises(ValueError, match="n_products"):
        SimulationConfig(n_products=0)

    with pytest.raises(ValueError, match="n_suppliers"):
        SimulationConfig(n_suppliers=-1)

    with pytest.raises(ValueError, match="episode_days"):
        SimulationConfig(episode_days=0)

    with pytest.raises(ValueError, match="transport_cost_per_unit"):
        SimulationConfig(transport_cost_per_unit=-0.5)


def test_product_dataclass_validation():
    prod = Product(
        id="PROD_001",
        name="Smartphone Case",
        category="Electronics",
        unit_cost=15.0,
        selling_price=35.0,
        holding_cost_rate=0.05,
        stockout_penalty=25.0,
        weight_kg=0.25,
        volume_m3=0.002,
    )
    assert prod.unit_margin == 20.0
    assert isinstance(prod.to_dict(), dict)
    assert prod.to_dict()["id"] == "PROD_001"

    # Negative cost check
    with pytest.raises(ValueError, match="unit_cost"):
        Product(
            id="PROD_BAD",
            name="Bad",
            unit_cost=-5.0,
            selling_price=10.0,
            holding_cost_rate=0.1,
            stockout_penalty=5.0,
        )

    # Zero or negative weight check
    with pytest.raises(ValueError, match="weight_kg"):
        Product(
            id="PROD_BAD",
            name="Bad",
            unit_cost=5.0,
            selling_price=10.0,
            holding_cost_rate=0.1,
            stockout_penalty=5.0,
            weight_kg=0.0,
        )


def test_warehouse_dataclass_validation():
    wh = Warehouse(
        id="WH_01",
        name="Main Fulfillment Hub",
        location="Chicago",
        storage_capacity=30000.0,
        daily_fixed_cost=450.0,
    )
    assert wh.storage_capacity == 30000.0
    assert wh.to_dict()["location"] == "Chicago"

    with pytest.raises(ValueError, match="storage_capacity"):
        Warehouse(id="WH_ERR", name="Err", storage_capacity=-100.0)


def test_supplier_dataclass_validation():
    sup = Supplier(
        id="SUP_01",
        name="Apex Manufacturing",
        base_lead_time_days=4,
        lead_time_std=1.2,
        reliability=0.96,
        fixed_order_cost=150.0,
        catalog={"PROD_001": 14.5},
    )
    assert sup.base_lead_time_days == 4
    assert sup.reliability == 0.96

    with pytest.raises(ValueError, match="base_lead_time_days"):
        Supplier(id="SUP_ERR", name="Err", base_lead_time_days=0)

    with pytest.raises(ValueError, match="reliability"):
        Supplier(id="SUP_ERR", name="Err", base_lead_time_days=2, reliability=1.5)


def test_disruption_event_dataclass_validation():
    disr = DisruptionEvent(
        id="DISR_TEST",
        name="Factory Strike",
        type=DisruptionType.SUPPLIER_DELAY,
        start_step=20,
        duration_steps=5,
        impact_multiplier=2.5,
    )
    assert not disr.is_active(19)
    assert disr.is_active(20)
    assert disr.is_active(24)
    assert not disr.is_active(25)

    with pytest.raises(ValueError, match="duration_steps"):
        DisruptionEvent(
            id="ERR",
            name="Err",
            type=DisruptionType.DEMAND_SPIKE,
            start_step=1,
            duration_steps=0,
        )


# --------------------------------------------------------------------------
# 2. Data Generator Network Topology (1 WH, 5 Products, 3 Suppliers)
# --------------------------------------------------------------------------

def test_generator_network_dimensions_phase1():
    cfg = get_default_config()
    gen = SupplyChainDataGenerator(cfg)
    products, warehouses, suppliers, disruptions, demand = gen.generate_supply_chain_network()

    assert len(warehouses) == 1
    assert len(products) == 5
    assert len(suppliers) == 3
    assert demand.shape == (90, 1, 5)
    assert len(disruptions) >= 1


def test_supplier_catalog_coverage_invariant():
    cfg = get_default_config()
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    suppliers = gen.generate_suppliers(products)

    all_pids = {p.id for p in products}
    covered_pids = {pid for s in suppliers for pid in s.catalog}
    assert all_pids.issubset(covered_pids), "Every product must have at least one valid supplier"


# --------------------------------------------------------------------------
# 3. Demand Properties: Non-negativity, Seasonality, Trend & Disruptions
# --------------------------------------------------------------------------

def test_demand_tensor_validity():
    cfg = get_default_config()
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    disruptions = gen.generate_disruptions(gen.generate_suppliers(products), warehouses, products)
    demand = gen.generate_demand_matrix(products, warehouses, disruptions)

    assert demand.dtype in [np.int32, np.int64, int]
    assert np.all(demand >= 0), "Demand must always be non-negative integer counts"
    assert not np.isnan(demand).any()


def test_demand_seasonality_and_trend():
    cfg = SimulationConfig(
        n_warehouses=1,
        n_products=5,
        n_suppliers=3,
        episode_days=90,
        random_seed=42,
        trend_slope=0.20,
        weekend_boost=0.30,
    )
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    demand = gen.generate_demand_matrix(products, warehouses, disruptions=[])

    # Test Weekend vs Weekday seasonality across the 90 days
    weekend_days = [t for t in range(90) if (t % 7) in [5, 6]]
    weekday_days = [t for t in range(90) if (t % 7) in [1, 2, 3]]  # Tuesday, Wed, Thu

    weekend_mean = float(np.mean(demand[weekend_days, 0, :]))
    weekday_mean = float(np.mean(demand[weekday_days, 0, :]))
    assert weekend_mean > weekday_mean, "Weekend demand should be systematically higher due to weekend_boost"

    # Test Positive Trend: Last 30 days should have higher average demand than first 30 days
    first_month_mean = float(np.mean(demand[0:30, 0, :]))
    last_month_mean = float(np.mean(demand[60:90, 0, :]))
    assert last_month_mean > first_month_mean, "Positive trend slope should increase demand over time"


def test_disruption_impact_on_demand_spike():
    cfg = get_default_config()
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()

    target_pid = products[0].id
    spike_disr = DisruptionEvent(
        id="SPIKE_TEST",
        name="Flash Sale Spike",
        type=DisruptionType.DEMAND_SPIKE,
        start_step=20,
        duration_steps=5,
        impact_multiplier=3.0,
        target_product_id=target_pid,
    )

    demand_with_spike = gen.generate_demand_matrix(products, warehouses, [spike_disr])

    # Compare step 20-24 vs step 10-14 for target product
    normal_demand = np.mean(demand_with_spike[10:15, 0, 0])
    spiked_demand = np.mean(demand_with_spike[20:25, 0, 0])
    assert spiked_demand > (normal_demand * 1.8), "Demand spike disruption must noticeably inflate demand"


# --------------------------------------------------------------------------
# 4. Stochastic Lead Times & Constraints
# --------------------------------------------------------------------------

def test_stochastic_lead_time_constraints():
    cfg = get_default_config()
    gen = SupplyChainDataGenerator(cfg)
    sup = Supplier(
        id="SUP_01",
        name="Test Supplier",
        base_lead_time_days=3,
        lead_time_std=1.5,
        reliability=0.90,
    )

    # Sample 500 lead times: all must be integer >= 1
    samples = [gen.sample_lead_time(sup, current_step=0) for _ in range(500)]
    assert all(isinstance(lt, int) for lt in samples)
    assert all(lt >= 1 for lt in samples), "Lead times must strictly be >= 1 day"
    assert 2.0 <= np.mean(samples) <= 4.5


def test_disruption_impact_on_lead_time():
    cfg = get_default_config()
    gen = SupplyChainDataGenerator(cfg)
    sup = Supplier(
        id="SUP_01",
        name="Bottleneck Supplier",
        base_lead_time_days=3,
        lead_time_std=0.5,
        reliability=0.99,
    )
    delay_disr = DisruptionEvent(
        id="DELAY_TEST",
        name="Supplier Bottleneck",
        type=DisruptionType.SUPPLIER_DELAY,
        start_step=10,
        duration_steps=10,
        impact_multiplier=2.5,
        target_supplier_id="SUP_01",
    )

    # Lead time during non-disrupted step
    normal_samples = [gen.sample_lead_time(sup, current_step=0, disruptions=[delay_disr]) for _ in range(200)]
    # Lead time during disrupted step (step 15)
    disrupted_samples = [gen.sample_lead_time(sup, current_step=15, disruptions=[delay_disr]) for _ in range(200)]

    assert np.mean(disrupted_samples) > np.mean(normal_samples) * 1.8


# --------------------------------------------------------------------------
# 5. Reproducibility
# --------------------------------------------------------------------------

def test_exact_reproducibility_with_fixed_seed():
    cfg1 = SimulationConfig(n_warehouses=1, n_products=5, n_suppliers=3, random_seed=42)
    cfg2 = SimulationConfig(n_warehouses=1, n_products=5, n_suppliers=3, random_seed=42)

    gen1 = SupplyChainDataGenerator(cfg1)
    gen2 = SupplyChainDataGenerator(cfg2)

    p1, w1, s1, d1, dem1 = gen1.generate_supply_chain_network()
    p2, w2, s2, d2, dem2 = gen2.generate_supply_chain_network()

    assert [p.unit_cost for p in p1] == [p.unit_cost for p in p2]
    assert [s.fixed_order_cost for s in s1] == [s.fixed_order_cost for s in s2]
    assert np.array_equal(dem1, dem2), "Demand matrix must be bit-for-bit identical with same seed"


def test_different_seeds_produce_different_demand():
    cfg1 = SimulationConfig(n_warehouses=1, n_products=5, n_suppliers=3, random_seed=42)
    cfg2 = SimulationConfig(n_warehouses=1, n_products=5, n_suppliers=3, random_seed=999)

    gen1 = SupplyChainDataGenerator(cfg1)
    gen2 = SupplyChainDataGenerator(cfg2)

    _, _, _, _, dem1 = gen1.generate_supply_chain_network()
    _, _, _, _, dem2 = gen2.generate_supply_chain_network()

    assert not np.array_equal(dem1, dem2), "Different seeds must produce different stochastic trajectories"
