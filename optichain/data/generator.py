"""
OptiChain AI - Multi-Echelon Data & Demand Generator
Generates realistic supply chain networks, catalogs, supplier lead time profiles,
and non-stationary customer demand time series with trend, seasonality, and disruptions.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from optichain.config import SimulationConfig
from optichain.core.entities import (
    Product,
    Warehouse,
    Supplier,
    DisruptionEvent,
    DisruptionType,
)


class SupplyChainDataGenerator:
    """
    Constructs a calibrated, realistic multi-echelon supply chain network
    and synthetic demand matrices. Default initialization: 1 WH, 5 Products, 3 Suppliers.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate_products(self) -> List[Product]:
        """
        Generates calibrated product SKUs with distinct cost, price, weight, volume,
        and inventory penalty profiles.
        """
        categories = ["Electronics", "Apparel", "FMCG", "HomeGoods", "Industrial"]
        products: List[Product] = []

        for i in range(self.config.n_products):
            pid = f"PROD_{i+1:03d}"
            cat = categories[i % len(categories)]

            if cat == "Electronics":
                unit_cost = round(float(self.rng.uniform(40.0, 180.0)), 2)
                margin = float(self.rng.uniform(0.35, 0.65))
                holding_rate = round(unit_cost * 0.005, 3)  # Daily carrying rate
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
        """
        Generates warehouse nodes with tiered storage capacities and operating costs.
        """
        warehouses: List[Warehouse] = []
        locations = [
            "North Region (Central Hub)",
            "West Region (Port Facility)",
            "South Region (Metro DC)",
        ]

        for i in range(self.config.n_warehouses):
            wid = f"WH_{i+1:02d}"
            loc = locations[i % len(locations)]
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
        """
        Generates suppliers with catalog pricing, minimum order quantities (MOQ),
        daily production capacities, and stochastic lead time profiles.
        Guarantees that every product SKU is covered by at least one supplier.
        """
        suppliers: List[Supplier] = []
        product_ids = [p.id for p in products]

        for s in range(self.config.n_suppliers):
            sid = f"SUP_{s+1:02d}"
            base_lead_time = int(self.rng.integers(2, 7))
            lead_std = round(float(self.rng.uniform(0.4, 1.4)), 2)
            reliability = round(float(self.rng.uniform(0.88, 0.99)), 2)
            fixed_cost = round(float(self.rng.uniform(60.0, 200.0)), 2)

            # Assign a subset of products to each supplier (or all if catalog is small)
            n_catalog = max(1, min(len(product_ids), int(len(product_ids) * float(self.rng.uniform(0.5, 1.0)))))
            chosen_pids = self.rng.choice(
                product_ids, size=min(n_catalog, len(product_ids)), replace=False
            )

            catalog: Dict[str, float] = {}
            moq: Dict[str, int] = {}
            daily_cap: Dict[str, int] = {}

            prod_map = {p.id: p for p in products}
            for pid in chosen_pids:
                p = prod_map[pid]
                supplier_price = round(p.unit_cost * float(self.rng.uniform(0.92, 1.08)), 2)
                catalog[pid] = supplier_price
                moq[pid] = int(self.rng.choice([1, 5, 10, 20]))
                daily_cap[pid] = int(self.rng.integers(100, 500))

            suppliers.append(
                Supplier(
                    id=sid,
                    name=f"Global Supplier {sid}",
                    location=f"Industrial Zone {(s % 3) + 1}",
                    base_lead_time_days=base_lead_time,
                    lead_time_std=lead_std,
                    reliability=reliability,
                    fixed_order_cost=fixed_cost,
                    catalog=catalog,
                    min_order_quantity=moq,
                    daily_capacity=daily_cap,
                )
            )

        # Invariant: Guarantee every product SKU has at least one active supplier
        covered_pids = {pid for sup in suppliers for pid in sup.catalog}
        uncovered = [pid for pid in product_ids if pid not in covered_pids]
        for pid in uncovered:
            sup = suppliers[self.rng.integers(0, len(suppliers))]
            p = [p for p in products if p.id == pid][0]
            sup.catalog[pid] = round(p.unit_cost, 2)
            sup.min_order_quantity[pid] = 5
            sup.daily_capacity[pid] = 200

        return suppliers

    def sample_lead_time(
        self,
        supplier: Supplier,
        current_step: int = 0,
        disruptions: Optional[List[DisruptionEvent]] = None,
    ) -> int:
        """
        Samples a stochastic lead time in days for a given supplier.
        Takes into account Gaussian variance, supplier delivery reliability,
        and active disruption multipliers (e.g. factory delays or port congestion).
        Enforces physical constraint: lead time >= 1 day.
        """
        base_lt = float(supplier.base_lead_time_days)
        std_lt = float(supplier.lead_time_std)
        raw_lt = self.rng.normal(base_lt, std_lt)

        # Reliability check: occasional delivery anomaly
        if self.rng.random() > supplier.reliability:
            raw_lt += float(self.rng.integers(1, 4))

        # Check disruption multipliers
        multiplier = 1.0
        if disruptions:
            for d in disruptions:
                if d.is_active(current_step):
                    if d.type == DisruptionType.SUPPLIER_DELAY and (
                        d.target_supplier_id is None or d.target_supplier_id == supplier.id
                    ):
                        multiplier = max(multiplier, d.impact_multiplier)
                    elif d.type == DisruptionType.PORT_CONGESTION:
                        multiplier = max(multiplier, d.impact_multiplier)

        effective_lt = max(1, int(round(raw_lt * multiplier)))
        return effective_lt

    def generate_disruptions(
        self,
        suppliers: List[Supplier],
        warehouses: List[Warehouse],
        products: List[Product],
    ) -> List[DisruptionEvent]:
        """
        Injects realistic, configurable supply chain disruptions across the episode horizon.
        """
        disruptions: List[DisruptionEvent] = []

        if self.config.episode_days >= 30:
            target_prod = products[0].id if products else None
            disruptions.append(
                DisruptionEvent(
                    id="DISR_01_DEMAND_SURGE",
                    name="Viral Product Demand Surge",
                    type=DisruptionType.DEMAND_SPIKE,
                    start_step=min(25, self.config.episode_days - 10),
                    duration_steps=7,
                    impact_multiplier=2.5,
                    target_product_id=target_prod,
                    description="Viral trend causes 250% demand surge for target product.",
                )
            )

        if self.config.episode_days >= 55:
            target_sup = suppliers[0].id if suppliers else None
            disruptions.append(
                DisruptionEvent(
                    id="DISR_02_SUPPLIER_DELAY",
                    name="Supplier Component Shortage",
                    type=DisruptionType.SUPPLIER_DELAY,
                    start_step=min(45, self.config.episode_days - 10),
                    duration_steps=10,
                    impact_multiplier=2.2,
                    target_supplier_id=target_sup,
                    description="Component bottleneck increases supplier lead time by 2.2x.",
                )
            )

        return disruptions

    def generate_demand_matrix(
        self,
        products: List[Product],
        warehouses: List[Warehouse],
        disruptions: Optional[List[DisruptionEvent]] = None,
    ) -> np.ndarray:
        """
        Generates non-stationary 3D customer demand tensor of shape (T, W, P).
        Incorporates:
        - Base mean demand per product category
        - Weekly cyclical seasonality (7-day periodic wave + weekend boost)
        - Long-term trend slope
        - Warehouse geographic scaling
        - Active disruption multipliers
        - Poisson stochastic noise (guaranteeing non-negative integer demand)
        """
        T = self.config.episode_days
        W = len(warehouses)
        P = len(products)
        disruptions = disruptions or []
        demand_matrix = np.zeros((T, W, P), dtype=int)

        # Base demand rates per product category
        base_rates = np.zeros(P)
        for p_idx, prod in enumerate(products):
            if prod.category == "FMCG":
                base_rates[p_idx] = float(self.rng.uniform(25.0, 60.0))
            elif prod.category == "Electronics":
                base_rates[p_idx] = float(self.rng.uniform(4.0, 15.0))
            else:
                base_rates[p_idx] = float(self.rng.uniform(10.0, 25.0))

        # Warehouse scaling
        wh_weights = np.array([1.0 + 0.10 * (w % 3) for w in range(W)])

        for t in range(T):
            # Day-of-week seasonality: peaked on weekend (dow 5, 6), trough mid-week (dow 2)
            dow = t % 7
            seasonality = 1.0 + self.config.seasonality_amplitude * math.cos(
                2.0 * math.pi * (dow - 5.5) / 7.0
            )
            if dow in [5, 6]:
                seasonality += self.config.weekend_boost

            # Long-term linear trend over the episode
            trend = 1.0 + self.config.trend_slope * (t / max(1, T))

            for w_idx in range(W):
                for p_idx, prod in enumerate(products):
                    mean_val = base_rates[p_idx] * wh_weights[w_idx] * seasonality * trend

                    # Check for active disruptions
                    for disr in disruptions:
                        if disr.is_active(t):
                            if disr.type == DisruptionType.DEMAND_SPIKE:
                                if (
                                    disr.target_product_id is None
                                    or disr.target_product_id == prod.id
                                ):
                                    mean_val *= disr.impact_multiplier

                    # Sample integer customer demand using Poisson distribution
                    demand = int(self.rng.poisson(lam=max(0.1, mean_val)))
                    demand_matrix[t, w_idx, p_idx] = demand

        return demand_matrix

    def generate_supply_chain_network(
        self,
    ) -> Tuple[List[Product], List[Warehouse], List[Supplier], List[DisruptionEvent], np.ndarray]:
        """
        Full convenience factory that constructs the complete network topology and demand tensor.
        """
        products = self.generate_products()
        warehouses = self.generate_warehouses()
        suppliers = self.generate_suppliers(products)
        disruptions = self.generate_disruptions(suppliers, warehouses, products)
        demand = self.generate_demand_matrix(products, warehouses, disruptions)
        return products, warehouses, suppliers, disruptions, demand
