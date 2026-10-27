"""
Unit tests for OptiChain AI synthetic data and demand generation.
"""

import numpy as np
from optichain.core.entities import SimulationConfig
from optichain.data.generator import SupplyChainDataGenerator


def test_generator_network_dimensions():
    cfg = SimulationConfig(
        n_warehouses=2,
        n_products=10,
        n_suppliers=4,
        episode_days=30,
        random_seed=123,
    )
    gen = SupplyChainDataGenerator(cfg)

    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    suppliers = gen.generate_suppliers(products)
    disruptions = gen.generate_disruptions(suppliers, warehouses, products)
    demand = gen.generate_demand_matrix(products, warehouses, disruptions)

    assert len(products) == 10
    assert len(warehouses) == 2
    assert len(suppliers) == 4
    assert demand.shape == (30, 2, 10)
    assert np.all(demand >= 0)
    assert len(disruptions) >= 1


def test_generator_supplier_catalog_coverage():
    cfg = SimulationConfig(
        n_warehouses=2,
        n_products=25,
        n_suppliers=5,
        episode_days=15,
        random_seed=999,
    )
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    suppliers = gen.generate_suppliers(products)

    # Invariant: Every product SKU must have at least one valid supplier
    all_pids = {p.id for p in products}
    covered_pids = {pid for sup in suppliers for pid in sup.catalog}
    assert all_pids.issubset(covered_pids)


def test_generator_reproducibility():
    cfg = SimulationConfig(n_warehouses=2, n_products=5, n_suppliers=3, episode_days=10, random_seed=42)
    gen1 = SupplyChainDataGenerator(cfg)
    gen2 = SupplyChainDataGenerator(cfg)

    prods1 = gen1.generate_products()
    prods2 = gen2.generate_products()
    assert [p.unit_cost for p in prods1] == [p.unit_cost for p in prods2]
