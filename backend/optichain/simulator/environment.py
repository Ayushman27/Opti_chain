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
        # 1. PROCESS ARRIVING DELIVERIES (Step 6)
        # ------------------------------------------------------------------
        still_in_transit: List[Order] = []
        arrived_shipments: List[Dict[str, Any]] = []

        for order in self.in_transit_orders:
            if order.expected_arrival_step <= t:
                order.status = OrderStatus.DELIVERED
                order.actual_arrival_step = t
                w_i = self.wh_idx[order.warehouse_id]
                p_i = self.prod_idx[order.product_id]

                self.on_hand[w_i, p_i] += order.quantity
                self.history_arrivals[t, w_i, p_i] += order.quantity
                new_arrived_units += int(order.quantity)

                prod_name = self.prod_map[order.product_id].name if order.product_id in self.prod_map else order.product_id
                sup_name = self.sup_map[order.supplier_id].name if order.supplier_id in self.sup_map else order.supplier_id
                arrived_shipments.append({
                    "order_id": order.id,
                    "product_id": order.product_id,
                    "product_name": prod_name,
                    "supplier_id": order.supplier_id,
                    "supplier_name": sup_name,
                    "warehouse_id": order.warehouse_id,
                    "quantity": int(order.quantity),
                })
            else:
                still_in_transit.append(order)
        self.in_transit_orders = still_in_transit

        # ------------------------------------------------------------------
        # 2. STATE SNAPSHOT & OBSERVATION (Step 7)
        # ------------------------------------------------------------------
        active_disruptions_today: List[Dict[str, Any]] = [
            {
                "type": d.type.value if hasattr(d.type, "value") else str(d.type),
                "name": getattr(d, "name", ""),
                "description": d.description,
                "severity": float(getattr(d, "impact_multiplier", 1.0)),
                "impact_multiplier": float(getattr(d, "impact_multiplier", 1.0)),
                "target_supplier": d.target_supplier_id,
                "target_warehouse": d.target_warehouse_id,
                "target_product": d.target_product_id,
            }
            for d in self.disruptions
            if d.is_active(t)
        ]
        pipeline_before_orders = sum(o.quantity for o in self.in_transit_orders if o.status == OrderStatus.IN_TRANSIT)
        on_hand_before_demand = float(np.sum(self.on_hand))
        effective_inv_before = on_hand_before_demand + pipeline_before_orders - float(np.sum(self.backlog))

        # ------------------------------------------------------------------
        # 3. PROCESS & VALIDATE NEW REPLENISHMENT ORDERS (Steps 8, 9, 10)
        # ------------------------------------------------------------------
        step_fixed_order_cost = 0.0
        step_procurement_cost = 0.0
        step_transport_cost = 0.0
        suppliers_ordered_from = set()
        supplier_dispatched_today: Dict[Tuple[str, str], int] = {}

        policy_requested_orders: List[Dict[str, Any]] = []
        placed_orders_detail: List[Dict[str, Any]] = []
        rejected_orders_detail: List[Dict[str, Any]] = []

        for (w_id, p_id, s_id), requested_qty in replenishment_orders.items():
            if requested_qty <= 0:
                continue

            policy_requested_orders.append({
                "warehouse_id": w_id,
                "product_id": p_id,
                "supplier_id": s_id,
                "requested_qty": int(requested_qty),
            })

            # Ensure valid entity references
            if w_id not in self.wh_map or p_id not in self.prod_map or s_id not in self.sup_map:
                rejected_orders_detail.append({
                    "product_id": p_id,
                    "supplier_id": s_id,
                    "reason": "Invalid entity reference (warehouse, product, or supplier not found)",
                })
                continue

            supplier = self.sup_map[s_id]
            product = self.prod_map[p_id]
            warehouse = self.wh_map[w_id]

            # Rule 1: Supplier Catalog Eligibility Check
            if p_id not in supplier.catalog:
                rejected_orders_detail.append({
                    "product_id": p_id,
                    "product_name": product.name,
                    "supplier_id": s_id,
                    "supplier_name": supplier.name,
                    "reason": f"SKU {product.name} ({p_id}) is not offered in {supplier.name} catalog",
                })
                continue

            unit_price = supplier.catalog[p_id]

            # Rule 2: Minimum Order Quantity (MOQ) Enforcement
            moq = supplier.min_order_quantity.get(p_id, 1)
            ordered_qty = max(requested_qty, moq)
            moq_applied = ordered_qty > requested_qty

            # Rule 3: Supplier Daily Production Capacity Check
            daily_cap = supplier.daily_capacity.get(p_id, 999999)
            already_dispatched = supplier_dispatched_today.get((s_id, p_id), 0)
            remaining_cap = max(0, daily_cap - already_dispatched)

            if remaining_cap <= 0:
                rejected_orders_detail.append({
                    "product_id": p_id,
                    "product_name": product.name,
                    "supplier_id": s_id,
                    "supplier_name": supplier.name,
                    "reason": f"Supplier {supplier.name} daily production capacity limit ({daily_cap}) fully exhausted",
                })
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
            order_id = f"ORD_{t:03d}_{self.order_counter:04d}"
            order = Order(
                id=order_id,
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

            placed_orders_detail.append({
                "order_id": order_id,
                "supplier_id": s_id,
                "supplier_name": supplier.name,
                "warehouse_id": w_id,
                "product_id": p_id,
                "product_name": product.name,
                "requested_qty": int(requested_qty),
                "final_qty": int(ordered_qty),
                "moq": int(moq),
                "moq_applied": bool(moq_applied),
                "daily_capacity": int(daily_cap),
                "lead_time_days": int(lead_time),
                "expected_arrival_day": int(arrival_step),
                "unit_cost": round(float(unit_price), 2),
                "line_procurement_cost": round(float(proc_cost), 2),
            })

        # ------------------------------------------------------------------
        # 4. CUSTOMER DEMAND REALIZATION & FULFILLMENT (Step 11)
        # ------------------------------------------------------------------
        step_demand_total = 0
        step_fulfilled_total = 0
        step_unmet_total = 0
        step_holding_cost = 0.0
        step_stockout_penalty = 0.0
        step_revenue = 0.0
        step_cogs = 0.0
        sku_breakdown: List[Dict[str, Any]] = []

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

                sku_breakdown.append({
                    "product_id": product.id,
                    "product_name": product.name,
                    "category": product.category,
                    "demand": int(customer_demand),
                    "fulfilled": int(fulfilled),
                    "unmet": int(unmet),
                    "stockout": unmet > 0,
                    "ending_on_hand": int(round(self.on_hand[w_i, p_i])),
                    "unit_cost": round(float(product.unit_cost), 2),
                    "selling_price": round(float(product.selling_price), 2),
                })

        # Warehouse fixed facility operating costs
        step_operating_cost = sum(w.daily_fixed_cost for w in self.warehouses)

        # ------------------------------------------------------------------
        # 5. FINANCIAL ACCOUNTING & TELEMETRY (Steps 12 & 13)
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
        pipeline_units_now = 0.0
        for o in self.in_transit_orders:
            if o.status == OrderStatus.IN_TRANSIT:
                w_idx = self.wh_idx[o.warehouse_id]
                p_idx = self.prod_idx[o.product_id]
                self.history_pipeline[t, w_idx, p_idx] += o.quantity
                pipeline_units_now += o.quantity

        self.metrics_tracker.record_step(
            demand=step_demand_total,
            fulfilled=step_fulfilled_total,
            unmet=step_unmet_total,
            on_hand_units=total_inventory_units,
            on_hand_value=total_inventory_val,
            cogs=step_cogs,
        )

        # Build detailed 8-step execution lifecycle matching Section 3.1 mechanics
        daily_execution_steps: List[Dict[str, Any]] = [
            {
                "step_num": 6,
                "title": "Inbound Shipment Arrival",
                "tag": "Arrivals",
                "status": "success" if arrived_shipments else "neutral",
                "summary": (
                    f"Received {len(arrived_shipments)} shipment(s) delivering {new_arrived_units:,} physical units into warehouse stock."
                    if arrived_shipments
                    else "No inbound shipments scheduled for delivery today."
                ),
                "items": arrived_shipments,
            },
            {
                "step_num": 7,
                "title": "Network State Observation",
                "tag": "Observation",
                "status": "warning" if active_disruptions_today else "info",
                "summary": (
                    f"On-Hand: {int(on_hand_before_demand):,} units | Pipeline: {int(pipeline_before_orders):,} units | "
                    f"Effective IP: {int(effective_inv_before):,} units | Disruptions: {len(active_disruptions_today)} active."
                ),
                "data": {
                    "on_hand": int(on_hand_before_demand),
                    "pipeline": int(pipeline_before_orders),
                    "effective_inventory": int(effective_inv_before),
                    "disruptions": active_disruptions_today,
                },
            },
            {
                "step_num": 8,
                "title": "Inventory Policy Decision",
                "tag": "Replenishment",
                "status": "info" if policy_requested_orders else "neutral",
                "summary": (
                    f"Inventory policy evaluated state and submitted {len(policy_requested_orders)} replenishment order request(s)."
                    if policy_requested_orders
                    else "Effective inventory position is above reorder thresholds; policy requested 0 orders today."
                ),
                "items": policy_requested_orders,
            },
            {
                "step_num": 9,
                "title": "Supplier Eligibility & MOQ Validation",
                "tag": "Constraints",
                "status": (
                    "warning"
                    if any(o["moq_applied"] for o in placed_orders_detail) or rejected_orders_detail
                    else "success"
                    if placed_orders_detail
                    else "neutral"
                ),
                "summary": (
                    f"Validated {len(placed_orders_detail)} order(s). "
                    + (
                        f"MOQ constraints elevated {sum(1 for o in placed_orders_detail if o['moq_applied'])} order(s). "
                        if any(o["moq_applied"] for o in placed_orders_detail)
                        else "All order quantities satisfied MOQ and daily capacity constraints. "
                    )
                    + (f"{len(rejected_orders_detail)} rejected." if rejected_orders_detail else "")
                ),
                "placed": placed_orders_detail,
                "rejected": rejected_orders_detail,
            },
            {
                "step_num": 10,
                "title": "Stochastic Lead Time & Pipeline Scheduling",
                "tag": "Lead Time",
                "status": "info" if placed_orders_detail else "neutral",
                "summary": (
                    f"Scheduled {len(placed_orders_detail)} accepted purchase order(s) for future arrival (enforcing physical lead time L >= 1 day)."
                    if placed_orders_detail
                    else "No new purchase orders added to in-transit pipeline today."
                ),
                "items": [
                    {
                        "order_id": o["order_id"],
                        "sku": o["product_name"],
                        "supplier": o["supplier_name"],
                        "quantity": o["final_qty"],
                        "lead_time": f"{o['lead_time_days']} day(s)",
                        "arrival_day": f"Day {o['expected_arrival_day']}",
                    }
                    for o in placed_orders_detail
                ],
            },
            {
                "step_num": 11,
                "title": "Customer Demand Realization & Fulfillment",
                "tag": "Fulfillment",
                "status": "error" if step_unmet_total > 0 else "success",
                "summary": (
                    f"Realized demand: {step_demand_total:,} units. Fulfilled {step_fulfilled_total:,} units ({step_unmet_total:,} unmet stockout units)."
                    if step_demand_total > 0
                    else "Zero customer demand realized today."
                ),
                "sku_breakdown": sku_breakdown,
            },
            {
                "step_num": 12,
                "title": "Daily Economic Balance Sheet",
                "tag": "Financials",
                "status": "success" if cost_breakdown.net_profit >= 0 else "warning",
                "summary": (
                    f"Daily Revenue: ₹{step_revenue:,.2f} | Total Costs: ₹{cost_breakdown.total_cost:,.2f} | "
                    f"Net Daily Profit: ₹{cost_breakdown.net_profit:,.2f}."
                ),
                "breakdown": {
                    "holding_cost": round(step_holding_cost, 2),
                    "stockout_penalty": round(step_stockout_penalty, 2),
                    "fixed_order_cost": round(step_fixed_order_cost, 2),
                    "procurement_cost": round(step_procurement_cost, 2),
                    "transport_cost": round(step_transport_cost, 2),
                    "operating_cost": round(step_operating_cost, 2),
                    "revenue": round(step_revenue, 2),
                    "net_profit": round(cost_breakdown.net_profit, 2),
                },
            },
            {
                "step_num": 13,
                "title": "Telemetry & Multi-Dimensional Ledger",
                "tag": "Telemetry",
                "status": "info",
                "summary": (
                    f"Ending On-Hand Stock: {int(total_inventory_units):,} units (₹{total_inventory_val:,.2f} asset value) | "
                    f"Daily Fill Rate: {(round(step_fulfilled_total / step_demand_total * 100, 1) if step_demand_total > 0 else 100.0)}%."
                ),
                "data": {
                    "ending_on_hand_units": round(total_inventory_units, 1),
                    "ending_on_hand_val": round(total_inventory_val, 2),
                    "pipeline_units": round(pipeline_units_now, 1),
                    "fill_rate": round(step_fulfilled_total / step_demand_total * 100, 2) if step_demand_total > 0 else 100.0,
                },
            },
        ]

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
            "active_disruptions": active_disruptions_today,
            "arrived_shipments": arrived_shipments,
            "placed_orders": placed_orders_detail,
            "rejected_orders": rejected_orders_detail,
            "sku_breakdown": sku_breakdown,
            "execution_steps": daily_execution_steps,
            "cost_breakdown_detail": {
                "holding": round(step_holding_cost, 2),
                "stockout": round(step_stockout_penalty, 2),
                "fixed_order": round(step_fixed_order_cost, 2),
                "procurement": round(step_procurement_cost, 2),
                "transport": round(step_transport_cost, 2),
                "operating": round(step_operating_cost, 2),
            },
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
