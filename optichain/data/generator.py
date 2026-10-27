"""
OptiChain AI - Multi-Echelon Data & Demand Generator
Generates realistic supply chain networks, catalogs, supplier lead time profiles,
and non-stationary customer demand time series with trend, seasonality, and disruptions.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np

from optichain.core.entities import (
    Product,
    Warehouse,
    Supplier,
    Disruption,
    DisruptionType,
    SimulationConfig,
)


class SupplyChainDataGenerator:
    """
    Constructs a calibrated, realistic multi-echelon supply chain network
    and synthetic demand matrices.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def generate_products(self) -> List[Product]:
        categories = ["Electronics", "Apparel", "FMCG", "HomeGoods", "Industrial"]
        products: List[Product] = []

        for i in range(self.config.n_products):
            pid = f"PROD_{i+1:03d}"
            cat = categories[i % len(categories)]
            
            # Category pricing and holding profiles
            if cat == "Electronics":
                unit_cost = round(float(self.rng.uniform(40.0, 180.0)), 2)
                margin = float(self.rng.uniform(0.35, 0.65))
                holding_rate = round(unit_cost * 0.005, 3)  # High holding cost
                stockout_penalty = round(unit_cost * 1.5, 2)
                weight = round(float(self.rng.uniform(0.5, 5.0)), 2)
            elif cat == "FMCG":
                unit_cost = round(float(self.rng.uniform(2.0, 15.0)), 2)
                margin = float(self.rng.uniform(0.20, 0.40))
                holding_rate = round(unit_cost * 0.002, 3)
                stockout_penalty = round(unit_cost * 1.2, 2)
                weight = round(float(self.rng.uniform(0.2, 2.0)), 2)
            elif cat == "Apparel":
                unit_cost = round(float(self.rng.uniform(10.0, 45.0)), 2)
                margin = float(self.rng.uniform(0.40, 0.70))
                holding_rate = round(unit_cost * 0.003, 3)
                stockout_penalty = round(unit_cost * 1.3, 2)
                weight = round(float(self.rng.uniform(0.3, 1.5)), 2)
            else:
                unit_cost = round(float(self.rng.uniform(8.0, 60.0)), 2)
                margin = float(self.rng.uniform(0.30, 0.50))
                holding_rate = round(unit_cost * 0.003, 3)
                stockout_penalty = round(unit_cost * 1.25, 2)
                weight = round(float(self.rng.uniform(1.0, 8.0)), 2)

            selling_price = round(unit_cost * (1.0 + margin), 2)

            product = Product(
                id=pid,
                name=f"{cat} Item #{i+1:03d}",
                category=cat,
                unit_cost=unit_cost,
                selling_price=selling_price,
                holding_cost_rate=holding_rate,
                stockout_penalty=stockout_penalty,
                weight_kg=weight,
                volume_m3=round(weight * 0.005, 4),
            )
            products.append(product)

        return products

    def generate_warehouses(self) -> List[Warehouse]:
        warehouses: List[Warehouse] = []
        locations = ["North Region (Central Hub)", "West Region (Port Facility)", "South Region (Metro Metro)"]

        for i in range(self.config.n_warehouses):
            wid = f"WH_{i+1:02d}"
            loc = locations[i % len(locations)]
            # Tiered capacities
            capacity = float(self.rng.integers(15000, 40000))
            warehouses.append(
                Warehouse(
                    id=wid,
                    name=f"Distribution Center {wid}",
                    location=loc,
                    storage_capacity=capacity,
                    daily_fixed_cost=round(float(self.rng.uniform(300.0, 750.0)), 2),
                )
            )
        return warehouses

    def generate_suppliers(self, products: List[Product]) -> List[Supplier]:
        suppliers: List[Supplier] = []
        product_ids = [p.id for p in products]

        for s in range(self.config.n_suppliers):
            sid = f"SUP_{s+1:02d}"
            base_lead_time = int(self.rng.integers(2, 8))
            lead_std = round(float(self.rng.uniform(0.4, 1.6)), 2)
            reliability = round(float(self.rng.uniform(0.85, 0.99)), 2)
            fixed_cost = round(float(self.rng.uniform(60.0, 220.0)), 2)

            # Assign a subset of products to each supplier (at least 30% of catalog)
            n_catalog = max(3, int(len(product_ids) * float(self.rng.uniform(0.3, 0.7))))
            chosen_pids = self.rng.choice(product_ids, size=min(n_catalog, len(product_ids)), replace=False)

            catalog: Dict[str, float] = {}
            moq: Dict[str, int] = {}
            daily_cap: Dict[str, int] = {}

            prod_map = {p.id: p for p in products}
            for pid in chosen_pids:
                p = prod_map[pid]
                # Price discount/premium (+- 10% around base cost)
                supplier_price = round(p.unit_cost * float(self.rng.uniform(0.90, 1.10)), 2)
                catalog[pid] = supplier_price
                moq[pid] = int(self.rng.choice([1, 5, 10, 25, 50]))
                daily_cap[pid] = int(self.rng.integers(100, 500))

            suppliers.append(
                Supplier(
                    id=sid,
                    name=f"Global Supplier {sid}",
                    location=f"Industrial Zone {(s % 4) + 1}",
                    base_lead_time_days=base_lead_time,
                    lead_time_std=lead_std,
                    reliability=reliability,
                    fixed_order_cost=fixed_cost,
                    catalog=catalog,
                    min_order_quantity=moq,
                    daily_capacity=daily_cap,
                )
            )

        # Guarantee every product has at least one supplier
        covered_pids = {pid for sup in suppliers for pid in sup.catalog}
        uncovered = [pid for pid in product_ids if pid not in covered_pids]
        for pid in uncovered:
            sup = suppliers[self.rng.integers(0, len(suppliers))]
            p = [p for p in products if p.id == pid][0]
            sup.catalog[pid] = round(p.unit_cost, 2)
            sup.min_order_quantity[pid] = 10
            sup.daily_capacity[pid] = 200

        return suppliers

    def generate_disruptions(
        self,
        suppliers: List[Supplier],
        warehouses: List[Warehouse],
        products: List[Product],
    ) -> List[Disruption]:
        """
        Injects realistic supply chain disruptions:
        - A demand surge (flash sale/pandemic spike) around day 25
        - A supplier delay event around day 45
        - A localized port/shipping congestion around day 70
        """
        disruptions: List[Disruption] = []

        if self.config.episode_days >= 30:
            target_prod = products[0].id if products else None
            disruptions.append(
                Disruption(
                    id="DISR_01_DEMAND_SURGE",
                    name="Viral Product Demand Surge",
                    type=DisruptionType.DEMAND_SPIKE,
                    start_step=min(25, self.config.episode_days - 10),
                    duration_steps=7,
                    impact_multiplier=2.8,
                    target_product_id=target_prod,
                    description="Viral media trend causes sudden 280% demand surge for target product.",
                )
            )

        if self.config.episode_days >= 60:
            target_sup = suppliers[0].id if suppliers else None
            disruptions.append(
                Disruption(
                    id="DISR_02_SUPPLIER_DELAY",
                    name="Raw Material Shortage & Factory Delay",
                    type=DisruptionType.SUPPLIER_DELAY,
                    start_step=min(45, self.config.episode_days - 10),
                    duration_steps=12,
                    impact_multiplier=2.5,
                    target_supplier_id=target_sup,
                    description="Key supplier experiences supply chain bottleneck; lead times increase 2.5x.",
                )
            )

        if self.config.episode_days >= 80:
            disruptions.append(
                Disruption(
                    id="DISR_03_PORT_CONGESTION",
                    name="Port Logistics Congestion",
                    type=DisruptionType.PORT_CONGESTION,
                    start_step=min(68, self.config.episode_days - 6),
                    duration_steps=8,
                    impact_multiplier=2.0,
                    description="Severe port congestion doubles lead times across all international inbound shipments.",
                )
            )

        return disruptions

    def generate_demand_matrix(
        self,
        products: List[Product],
        warehouses: List[Warehouse],
        disruptions: List[Disruption],
    ) -> np.ndarray:
        """
        Generates 3D demand tensor of shape (n_days, n_warehouses, n_products)
        Incorporate non-stationary characteristics:
        - Base mean demand per product
        - Day-of-week seasonality (7-day periodic wave)
        - Monthly trend
        - Warehouse capacity scaling
        - Poisson / Log-normal random fluctuations
        - Active disruptions
        """
        T = self.config.episode_days
        W = len(warehouses)
        P = len(products)
        demand_matrix = np.zeros((T, W, P), dtype=int)

        # Base demand rate per product (Fast moving: 20-60, Slow moving: 2-10)
        base_rates = np.zeros(P)
        for p_idx, prod in enumerate(products):
            if prod.category == "FMCG":
                base_rates[p_idx] = float(self.rng.uniform(25.0, 70.0))
            elif prod.category == "Electronics":
                base_rates[p_idx] = float(self.rng.uniform(3.0, 18.0))
            else:
                base_rates[p_idx] = float(self.rng.uniform(8.0, 30.0))

        # Warehouse geographical demand multipliers
        wh_weights = np.array([1.0 + 0.15 * (w % 3) for w in range(W)])

        for t in range(T):
            # Day-of-week seasonality: Weekend boost
            dow = t % 7
            seasonality = 1.0 + 0.25 * math.sin(2.0 * math.pi * dow / 7.0)
            if dow in [5, 6]:
                seasonality += 0.20  # Weekend boost

            # Slight quarterly trend (e.g. +10% over 90 days)
            trend = 1.0 + 0.12 * (t / max(1, T))

            for w_idx in range(W):
                for p_idx, prod in enumerate(products):
                    mean_val = base_rates[p_idx] * wh_weights[w_idx] * seasonality * trend

                    # Check for active disruptions targeting this product
                    for disr in disruptions:
                        if disr.is_active(t):
                            if disr.type == DisruptionType.DEMAND_SPIKE:
                                if disr.target_product_id is None or disr.target_product_id == prod.id:
                                    mean_val *= disr.impact_multiplier

                    # Sample from Poisson or negative binomial for integer demand
                    demand = int(self.rng.poisson(lam=max(0.1, mean_val)))
                    demand_matrix[t, w_idx, p_idx] = demand

        return demand_matrix
