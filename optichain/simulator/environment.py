"""
OptiChain AI - Digital Twin Simulation Engine
Provides a high-fidelity discrete-time multi-echelon supply chain simulator
with stochastic lead times, disruption scenarios, inventory pipeline tracking,
and rigorous financial accounting.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

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
from optichain.core.cost_engine import CostEngine, CostBreakdown
from optichain.simulator.metrics import MetricsTracker, EpisodeKPIs
from optichain.data.generator import SupplyChainDataGenerator


class DigitalTwinSimulator:
    """
    Digital Twin Supply Chain Environment.
    Simulates inventory transitions, supply delays, demand shocks,
    and granular financial balance sheets across multi-echelon networks.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        products: Optional[List[Product]] = None,
        warehouses: Optional[List[Warehouse]] = None,
        suppliers: Optional[List[Supplier]] = None,
        disruptions: Optional[List[Disruption]] = None,
        demand_matrix: Optional[np.ndarray] = None,
    ):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        # Initialize or generate network entities
        if products is None or warehouses is None or suppliers is None:
            generator = SupplyChainDataGenerator(self.config)
            self.products = generator.generate_products()
            self.warehouses = generator.generate_warehouses()
            self.suppliers = generator.generate_suppliers(self.products)
            self.disruptions = (
                disruptions
                if disruptions is not None
                else generator.generate_disruptions(
                    self.suppliers, self.warehouses, self.products
                )
            )
            self.demand_matrix = (
                demand_matrix
                if demand_matrix is not None
                else generator.generate_demand_matrix(
                    self.products, self.warehouses, self.disruptions
                )
            )
        else:
            self.products = products
            self.warehouses = warehouses
            self.suppliers = suppliers
            self.disruptions = disruptions or []
            if demand_matrix is None:
                generator = SupplyChainDataGenerator(self.config)
                self.demand_matrix = generator.generate_demand_matrix(
                    self.products, self.warehouses, self.disruptions
                )
            else:
                self.demand_matrix = demand_matrix

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
        self.current_step: int = 0
        self.order_counter: int = 0
        self.on_hand: np.ndarray = np.zeros(
            (len(self.warehouses), len(self.products)), dtype=float
        )
        self.backlog: np.ndarray = np.zeros(
            (len(self.warehouses), len(self.products)), dtype=float
        )
        self.in_transit_orders: List[Order] = []
        self.all_orders_history: List[Order] = []
        self.step_history: List[Dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Resets simulator state to day 0 with warm-up initial inventory.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.order_counter = 0
        self.cost_engine.reset()
        self.metrics_tracker.reset()
        self.in_transit_orders.clear()
        self.all_orders_history.clear()
        self.step_history.clear()

        # Initialize on-hand inventory (e.g. 7-10 days of average demand safety stock)
        W = len(self.warehouses)
        P = len(self.products)
        self.on_hand = np.zeros((W, P), dtype=float)
        self.backlog = np.zeros((W, P), dtype=float)

        mean_demand = self.demand_matrix.mean(axis=0)  # Shape (W, P)
        for w in range(W):
            for p in range(P):
                # Warm-up buffer of ~8 days
                init_stock = max(10.0, float(mean_demand[w, p] * 8.0))
                self.on_hand[w, p] = round(init_stock, 1)

        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """
        Returns structured snapshot of the supply chain state.
        """
        W = len(self.warehouses)
        P = len(self.products)

        # Calculate pipeline inventory (units on order) per warehouse and product
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

        return {
            "current_step": self.current_step,
            "on_hand": self.on_hand.copy(),
            "in_transit": pipeline,
            "effective_inventory": self.on_hand + pipeline - self.backlog,
            "backlog": self.backlog.copy(),
            "active_disruptions": active_disruptions,
            "is_terminal": self.current_step >= self.config.episode_days,
        }

    def _sample_lead_time(self, supplier: Supplier, current_step: int) -> int:
        """
        Samples stochastic lead time influenced by active disruptions.
        """
        base_lt = float(supplier.base_lead_time_days)
        std_lt = float(supplier.lead_time_std)
        raw_lt = self.rng.normal(base_lt, std_lt)

        # Check reliability: occasional delivery anomaly
        if self.rng.random() > supplier.reliability:
            raw_lt += float(self.rng.integers(2, 5))

        # Check disruption multipliers
        multiplier = 1.0
        for d in self.disruptions:
            if d.is_active(current_step):
                if d.type == DisruptionType.SUPPLIER_DELAY and (
                    d.target_supplier_id is None or d.target_supplier_id == supplier.id
                ):
                    multiplier = max(multiplier, d.impact_multiplier)
                elif d.type == DisruptionType.PORT_CONGESTION:
                    multiplier = max(multiplier, d.impact_multiplier)

        effective_lt = max(1, int(round(raw_lt * multiplier)))
        return effective_lt

    def step(
        self, replenishment_orders: Dict[Tuple[str, str, str], int]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one discrete simulation day:
        1. Receive arrived shipments from in-transit pipeline.
        2. Issue new replenishment orders to suppliers.
        3. Fulfill customer demand and record stockouts.
        4. Tally costs and update metrics.

        `replenishment_orders` maps (warehouse_id, product_id, supplier_id) -> quantity
        """
        if self.current_step >= self.config.episode_days:
            raise RuntimeError("Simulation episode already terminated. Call reset() first.")

        t = self.current_step
        new_arrived_units = 0

        # ----------------------------------------------------
        # 1. PROCESS ARRIVING ORDERS
        # ----------------------------------------------------
        still_in_transit: List[Order] = []
        for order in self.in_transit_orders:
            if order.expected_arrival_step <= t:
                order.status = OrderStatus.DELIVERED
                order.actual_arrival_step = t
                w_i = self.wh_idx[order.warehouse_id]
                p_i = self.prod_idx[order.product_id]
                self.on_hand[w_i, p_i] += order.quantity
                new_arrived_units += order.quantity
            else:
                still_in_transit.append(order)
        self.in_transit_orders = still_in_transit

        # ----------------------------------------------------
        # 2. PROCESS NEW REPLENISHMENT ORDERS
        # ----------------------------------------------------
        step_fixed_order_cost = 0.0
        step_procurement_cost = 0.0
        step_transport_cost = 0.0
        suppliers_ordered_from = set()

        for (w_id, p_id, s_id), qty in replenishment_orders.items():
            if qty <= 0:
                continue

            supplier = self.sup_map[s_id]
            product = self.prod_map[p_id]
            warehouse = self.wh_map[w_id]

            # Enforce supplier catalog and MOQ
            unit_price = supplier.catalog.get(p_id, product.unit_cost)
            moq = supplier.min_order_quantity.get(p_id, 1)
            ordered_qty = max(qty, moq) if qty > 0 else 0

            if ordered_qty == 0:
                continue

            # Daily capacity limit check
            daily_cap = supplier.daily_capacity.get(p_id, 999999)
            ordered_qty = min(ordered_qty, daily_cap)

            # Sample lead time with disruption logic
            lead_time = self._sample_lead_time(supplier, t)
            arrival_step = t + lead_time

            # Costs
            transport_cost = ordered_qty * self.config.transport_cost_per_unit
            proc_cost = ordered_qty * unit_price
            step_procurement_cost += proc_cost
            step_transport_cost += transport_cost

            # Fixed order cost charged once per supplier active in this step
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

        # ----------------------------------------------------
        # 3. FULFILL DEMAND & RECORD STOCKOUTS
        # ----------------------------------------------------
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

                available = self.on_hand[w_i, p_i]
                fulfilled = min(available, effective_demand)
                unmet = effective_demand - fulfilled

                self.on_hand[w_i, p_i] = max(0.0, available - fulfilled)

                if self.config.backlog_unmet_demand:
                    self.backlog[w_i, p_i] = unmet
                else:
                    self.backlog[w_i, p_i] = 0.0  # Lost sales

                step_fulfilled_total += int(fulfilled)
                step_unmet_total += int(unmet)

                # Economic calculation
                step_revenue += fulfilled * product.selling_price
                step_cogs += fulfilled * product.unit_cost
                step_holding_cost += self.on_hand[w_i, p_i] * product.holding_cost_rate
                step_stockout_penalty += unmet * product.stockout_penalty

        # Warehouse fixed facility costs
        step_operating_cost = sum(w.daily_fixed_cost for w in self.warehouses)

        # ----------------------------------------------------
        # 4. RECORD FINANCIALS & TELEMETRY
        # ----------------------------------------------------
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
                for w_i in range(len(self.warehouses))
                for p_i in range(len(self.products))
            )
        )

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

        # RL Reward formulation: negative total cost (or net profit)
        reward = float(-cost_breakdown.total_cost)

        return self.get_observation(), reward, terminated, truncated, step_info

    def get_kpis(self) -> EpisodeKPIs:
        return self.metrics_tracker.compute_kpis(self.cost_engine.get_summary())
