"""
OptiChain AI - Digital Twin Simulation Engine
Provides a high-fidelity discrete-time multi-echelon supply chain simulator
with stochastic lead times, disruption scenarios, inventory pipeline tracking,
supplier capacity & MOQ constraints, and rigorous financial accounting.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from optichain.config import SimulationConfig
from optichain.core.entities import (
    Product,
    Warehouse,
    Supplier,
    Order,
    OrderStatus,
    DisruptionEvent,
    DisruptionType,
)
from optichain.core.cost_engine import CostEngine, CostBreakdown
from optichain.simulator.metrics import MetricsTracker, EpisodeKPIs
from optichain.data.generator import SupplyChainDataGenerator


class DigitalTwinSimulator:
    """
    Digital Twin Supply Chain Environment.

    Discrete-Time Simulation Event Ordering per Step t:
    --------------------------------------------------
    1. ARRIVALS: Orders in transit with expected_arrival_step <= t are received into on-hand stock.
    2. REPLENISHMENT: New replenishment purchase orders placed by the policy are validated against
       supplier catalog eligibility, minimum order quantity (MOQ), and daily capacity limits.
       Stochastic lead time L >= 1 is sampled; orders are scheduled for future arrival at (t + L).
    3. DEMAND REALIZATION & FULFILLMENT: Customer demand D_{w,p,t} arrives and is fulfilled from
       available on-hand stock: F = min(I, D). Unmet demand U = D - F is recorded as stockouts.
       On-hand inventory is updated: I_{t+1} = I_t - F (strictly non-negative).
    4. COST ACCOUNTING & TELEMETRY: Holding costs on ending inventory, stockout penalties, PO setup,
       procurement, transportation, and facility operating costs are tallied in the CostEngine.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        products: Optional[List[Product]] = None,
        warehouses: Optional[List[Warehouse]] = None,
        suppliers: Optional[List[Supplier]] = None,
        disruptions: Optional[List[DisruptionEvent]] = None,
        demand_matrix: Optional[np.ndarray] = None,
        historical_demand_matrix: Optional[np.ndarray] = None,
    ):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.data_generator = SupplyChainDataGenerator(self.config)

        # Initialize or generate network entities
        if products is None or warehouses is None or suppliers is None:
            self.products = self.data_generator.generate_products()
            self.warehouses = self.data_generator.generate_warehouses()
            self.suppliers = self.data_generator.generate_suppliers(self.products)
            self.disruptions = (
                disruptions
                if disruptions is not None
                else self.data_generator.generate_disruptions(
                    self.suppliers, self.warehouses, self.products
                )
            )
            self.demand_matrix = (
                demand_matrix
                if demand_matrix is not None
                else self.data_generator.generate_demand_matrix(
                    self.products, self.warehouses, self.disruptions
                )
            )
        else:
            self.products = products
            self.warehouses = warehouses
            self.suppliers = suppliers
            self.disruptions = disruptions or []
            if demand_matrix is None:
                self.demand_matrix = self.data_generator.generate_demand_matrix(
                    self.products, self.warehouses, self.disruptions
                )
            else:
                self.demand_matrix = demand_matrix

        # Generate or assign separate historical demand strictly prior to simulation horizon (t < 0)
        if historical_demand_matrix is not None:
            self.historical_demand_matrix = historical_demand_matrix
        else:
            hist_generator = SupplyChainDataGenerator(
                self.config.model_copy(update={"random_seed": self.config.random_seed + 10000})
            )
            self.historical_demand_matrix = hist_generator.generate_demand_matrix(
                self.products, self.warehouses, disruptions=[]
            )

        # Lookups
        self.prod_map = {p.id: p for p in self.products}
        self.wh_map = {w.id: w for w in self.warehouses}
        self.sup_map = {s.id: s for s in self.suppliers}
        self.prod_idx = {p.id: i for i, p in enumerate(self.products)}
        self.wh_idx = {w.id: i for i, w in enumerate(self.warehouses)}

        # Financial & telemetry engines
        self.cost_engine = CostEngine()
        self.metrics_tracker = MetricsTracker()

        # State containers
        W = len(self.warehouses)
        P = len(self.products)
        T = self.config.episode_days

        self.current_step: int = 0
        self.order_counter: int = 0
        self.on_hand: np.ndarray = np.zeros((W, P), dtype=float)
        self.backlog: np.ndarray = np.zeros((W, P), dtype=float)
        self.in_transit_orders: List[Order] = []
        self.all_orders_history: List[Order] = []
        self.step_history: List[Dict[str, Any]] = []

        # Granular multi-dimensional historical tracking tensors: (T, W, P)
        self.history_on_hand: np.ndarray = np.zeros((T, W, P), dtype=float)
        self.history_demand: np.ndarray = np.zeros((T, W, P), dtype=float)
        self.history_fulfilled: np.ndarray = np.zeros((T, W, P), dtype=float)
        self.history_unmet: np.ndarray = np.zeros((T, W, P), dtype=float)
        self.history_arrivals: np.ndarray = np.zeros((T, W, P), dtype=float)
        self.history_pipeline: np.ndarray = np.zeros((T, W, P), dtype=float)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Resets simulator state to day 0 with deterministic warm-up initial inventory.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.data_generator = SupplyChainDataGenerator(
                self.config.model_copy(update={"random_seed": seed})
            )

        self.current_step = 0
        self.order_counter = 0
        self.cost_engine.reset()
        self.metrics_tracker.reset()
        self.in_transit_orders.clear()
        self.all_orders_history.clear()
        self.step_history.clear()

        W = len(self.warehouses)
        P = len(self.products)
        T = self.config.episode_days

        self.on_hand = np.zeros((W, P), dtype=float)
        self.backlog = np.zeros((W, P), dtype=float)

        self.history_on_hand.fill(0.0)
        self.history_demand.fill(0.0)
        self.history_fulfilled.fill(0.0)
        self.history_unmet.fill(0.0)
        self.history_arrivals.fill(0.0)
        self.history_pipeline.fill(0.0)

        # Warm-up initial inventory (e.g. ~7 days of average demand buffer as exact integer)
        mean_demand = self.demand_matrix.mean(axis=0)  # Shape (W, P)
        for w in range(W):
            for p in range(P):
                init_stock = max(10, int(round(mean_demand[w, p] * 7.0)))
                self.on_hand[w, p] = float(init_stock)

        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """
        Returns structured state observation snapshot including on-hand inventory,
        in-transit pipelines, effective inventory positions, and active disruptions.
        """
        W = len(self.warehouses)
        P = len(self.products)

        pipeline = np.zeros((W, P), dtype=float)
        for o in self.in_transit_orders:
            if o.status == OrderStatus.IN_TRANSIT:
                w_i = self.wh_idx[o.warehouse_id]
                p_i = self.prod_idx[o.product_id]
                pipeline[w_i, p_i] += o.quantity

        active_disruptions = [
            d.model_dump()
            for d in self.disruptions
            if d.is_active(self.current_step)
        ]

        # Inventory Position = OnHand + InTransit - Backlog
        effective_inventory = self.on_hand + pipeline - self.backlog

        return {
            "current_step": self.current_step,
            "on_hand": self.on_hand.copy(),
            "in_transit": pipeline,
            "effective_inventory": effective_inventory,
            "backlog": self.backlog.copy(),
            "active_disruptions": active_disruptions,
            "is_terminal": self.current_step >= self.config.episode_days,
        }

    def _sample_lead_time(self, supplier: Supplier, current_step: int) -> int:
        """
        Samples stochastic supplier delivery lead time adhering to Phase 1 data foundation.
        Enforces physical lower bound: L >= 1 day.
        """
        return self.data_generator.sample_lead_time(
            supplier=supplier,
            current_step=current_step,
            disruptions=self.disruptions,
        )

    def step(
        self, replenishment_orders: Dict[Tuple[str, str, str], int]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one discrete simulation day:
        1. Process arriving in-transit shipments into on-hand inventory.
        2. Validate & schedule new replenishment orders (catalog eligibility, MOQ, capacity limits).
        3. Realize customer demand and fulfill from available on-hand stock.
        4. Tally economic balance sheet and record telemetry.
        """
        if self.current_step >= self.config.episode_days:
            raise RuntimeError("Simulation episode already terminated. Call reset() first.")

        t = self.current_step
        W = len(self.warehouses)
        P = len(self.products)
        new_arrived_units = 0

        # ------------------------------------------------------------------
        # 1. PROCESS ARRIVING DELIVERIES
        # ------------------------------------------------------------------
        still_in_transit: List[Order] = []
        for order in self.in_transit_orders:
            if order.expected_arrival_step <= t:
                order.status = OrderStatus.DELIVERED
                order.actual_arrival_step = t
                w_i = self.wh_idx[order.warehouse_id]
                p_i = self.prod_idx[order.product_id]

                self.on_hand[w_i, p_i] += order.quantity
                self.history_arrivals[t, w_i, p_i] += order.quantity
                new_arrived_units += order.quantity
            else:
                still_in_transit.append(order)
        self.in_transit_orders = still_in_transit

        # ------------------------------------------------------------------
        # 2. PROCESS & VALIDATE NEW REPLENISHMENT ORDERS
        # ------------------------------------------------------------------
        step_fixed_order_cost = 0.0
        step_procurement_cost = 0.0
        step_transport_cost = 0.0
        suppliers_ordered_from = set()
        supplier_dispatched_today: Dict[Tuple[str, str], int] = {}

        for (w_id, p_id, s_id), requested_qty in replenishment_orders.items():
            if requested_qty <= 0:
                continue

            # Ensure valid entity references
            if w_id not in self.wh_map or p_id not in self.prod_map or s_id not in self.sup_map:
                continue

            supplier = self.sup_map[s_id]
            product = self.prod_map[p_id]
            warehouse = self.wh_map[w_id]

            # Rule 1: Supplier Catalog Eligibility Check
            if p_id not in supplier.catalog:
                # Supplier does not produce or offer this SKU
                continue

            unit_price = supplier.catalog[p_id]

            # Rule 2: Minimum Order Quantity (MOQ) Enforcement
            moq = supplier.min_order_quantity.get(p_id, 1)
            ordered_qty = max(requested_qty, moq)

            # Rule 3: Supplier Daily Production Capacity Check
            daily_cap = supplier.daily_capacity.get(p_id, 999999)
            already_dispatched = supplier_dispatched_today.get((s_id, p_id), 0)
            remaining_cap = max(0, daily_cap - already_dispatched)

            if remaining_cap <= 0:
                continue

            ordered_qty = min(ordered_qty, remaining_cap)
            supplier_dispatched_today[(s_id, p_id)] = already_dispatched + ordered_qty

            # Rule 4: Stochastic Lead Time Sampling (L >= 1)
            lead_time = self._sample_lead_time(supplier, t)
            arrival_step = t + lead_time  # Arrives strictly in future step >= t + 1

            # Costs
            transport_cost = ordered_qty * self.config.transport_cost_per_unit
            proc_cost = ordered_qty * unit_price
            step_procurement_cost += proc_cost
            step_transport_cost += transport_cost

            # Fixed order setup cost (charged once per supplier receiving at least one PO today)
            fixed_cost = 0.0
            if s_id not in suppliers_ordered_from:
                fixed_cost = supplier.fixed_order_cost
                step_fixed_order_cost += fixed_cost
                suppliers_ordered_from.add(s_id)

            self.order_counter += 1
            order = Order(
                id=f"ORD_{t:03d}_{self.order_counter:04d}",
                step_created=t,
                supplier_id=s_id,
                warehouse_id=w_id,
                product_id=p_id,
                quantity=ordered_qty,
                unit_cost=unit_price,
                fixed_order_cost=fixed_cost,
                transport_cost=transport_cost,
                expected_arrival_step=arrival_step,
                status=OrderStatus.IN_TRANSIT,
            )
            self.in_transit_orders.append(order)
            self.all_orders_history.append(order)

        # ------------------------------------------------------------------
        # 3. CUSTOMER DEMAND REALIZATION & FULFILLMENT
        # ------------------------------------------------------------------
        step_demand_total = 0
        step_fulfilled_total = 0
        step_unmet_total = 0
        step_holding_cost = 0.0
        step_stockout_penalty = 0.0
        step_revenue = 0.0
        step_cogs = 0.0

        for w_i, warehouse in enumerate(self.warehouses):
            for p_i, product in enumerate(self.products):
                customer_demand = int(self.demand_matrix[t, w_i, p_i])
                step_demand_total += customer_demand

                effective_demand = customer_demand
                if self.config.backlog_unmet_demand:
                    effective_demand += int(self.backlog[w_i, p_i])

                available = int(round(self.on_hand[w_i, p_i]))
                fulfilled = min(available, effective_demand)
                unmet = effective_demand - fulfilled

                # Update on-hand inventory: strictly non-negative
                self.on_hand[w_i, p_i] = float(max(0, available - fulfilled))

                if self.config.backlog_unmet_demand:
                    self.backlog[w_i, p_i] = float(unmet)
                else:
                    self.backlog[w_i, p_i] = 0.0  # Lost sales

                step_fulfilled_total += fulfilled
                step_unmet_total += unmet

                # Record granular SKU history tensors
                self.history_demand[t, w_i, p_i] = customer_demand
                self.history_fulfilled[t, w_i, p_i] = fulfilled
                self.history_unmet[t, w_i, p_i] = unmet
                self.history_on_hand[t, w_i, p_i] = self.on_hand[w_i, p_i]

                # Economics
                step_revenue += fulfilled * product.selling_price
                step_cogs += fulfilled * product.unit_cost
                step_holding_cost += self.on_hand[w_i, p_i] * product.holding_cost_rate
                step_stockout_penalty += unmet * product.stockout_penalty

        # Warehouse fixed facility operating costs
        step_operating_cost = sum(w.daily_fixed_cost for w in self.warehouses)

        # ------------------------------------------------------------------
        # 4. FINANCIAL ACCOUNTING & TELEMETRY
        # ------------------------------------------------------------------
        cost_breakdown = self.cost_engine.record_step_cost(
            step=t,
            holding_cost=step_holding_cost,
            stockout_penalty=step_stockout_penalty,
            fixed_order_cost=step_fixed_order_cost,
            procurement_cost=step_procurement_cost,
            transport_cost=step_transport_cost,
            operating_cost=step_operating_cost,
            revenue=step_revenue,
        )

        total_inventory_units = float(np.sum(self.on_hand))
        total_inventory_val = float(
            sum(
                self.on_hand[w_i, p_i] * self.products[p_i].unit_cost
                for w_i in range(W)
                for p_i in range(P)
            )
        )

        # Record pipeline snapshot in tensor
        for o in self.in_transit_orders:
            if o.status == OrderStatus.IN_TRANSIT:
                w_idx = self.wh_idx[o.warehouse_id]
                p_idx = self.prod_idx[o.product_id]
                self.history_pipeline[t, w_idx, p_idx] += o.quantity

        self.metrics_tracker.record_step(
            demand=step_demand_total,
            fulfilled=step_fulfilled_total,
            unmet=step_unmet_total,
            on_hand_units=total_inventory_units,
            on_hand_value=total_inventory_val,
            cogs=step_cogs,
        )

        step_info = {
            "step": t,
            "demand": step_demand_total,
            "fulfilled": step_fulfilled_total,
            "unmet": step_unmet_total,
            "fill_rate": (
                round(step_fulfilled_total / step_demand_total * 100, 2)
                if step_demand_total > 0
                else 100.0
            ),
            "total_cost": round(cost_breakdown.total_cost, 2),
            "revenue": round(step_revenue, 2),
            "net_profit": round(cost_breakdown.net_profit, 2),
            "on_hand_units": round(total_inventory_units, 1),
            "arrived_units": new_arrived_units,
            "orders_placed_count": len(replenishment_orders),
        }
        self.step_history.append(step_info)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.config.episode_days
        truncated = False
        reward = float(-cost_breakdown.total_cost)

        return self.get_observation(), reward, terminated, truncated, step_info

    def get_kpis(self) -> EpisodeKPIs:
        return self.metrics_tracker.compute_kpis(self.cost_engine.get_summary())
