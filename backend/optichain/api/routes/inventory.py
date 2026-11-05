"""
Inventory route
GET /api/v1/inventory
"""

import math
from fastapi import APIRouter
import numpy as np

from optichain.api.schemas import InventoryResponse, ProductInventoryItem
from optichain.api.dependencies import get_reference_simulation
from optichain.baselines.rop_policy import ROPPolicy

router = APIRouter(tags=["Inventory"])

_cached_inventory_response = None


@router.get("/inventory", response_model=InventoryResponse)
def get_inventory() -> InventoryResponse:
    """
    Returns product-level inventory status, pipeline in-transit orders,
    fulfilled and unmet demand, and policy reorder thresholds from the digital twin simulator.
    """
    global _cached_inventory_response
    if _cached_inventory_response is not None:
        return _cached_inventory_response

    sim = get_reference_simulation(seed=42)
    sim.reset(seed=42)

    # Run for 14 operational days using standard ROP policy to establish live dynamics
    policy = ROPPolicy(service_factor_z=1.65)
    policy.fit(sim)

    for step in range(14):
        obs = sim.get_observation()
        action = policy.get_actions(obs, sim)
        sim.step(action)

    w_idx = 0  # Single primary distribution center in benchmark configuration
    wh = sim.warehouses[w_idx]

    products_list = []
    total_on_hand = 0.0
    total_pipeline = 0.0
    total_fulfilled = 0.0
    total_unmet = 0.0

    for p_idx, prod in enumerate(sim.products):
        on_hand = float(sim.on_hand[w_idx, p_idx])
        total_on_hand += on_hand

        # In-transit orders for this SKU
        pipeline = sum(
            float(order.quantity)
            for order in sim.in_transit_orders
            if order.product_id == prod.id and order.warehouse_id == wh.id
        )
        total_pipeline += pipeline

        # Fulfilled and unmet totals
        fulfilled = float(np.sum(sim.history_fulfilled[:14, w_idx, p_idx]))
        unmet = float(np.sum(sim.history_unmet[:14, w_idx, p_idx]))
        total_fulfilled += fulfilled
        total_unmet += unmet

        # Recent daily demand rate (last 7 days)
        recent_demands = sim.history_demand[max(0, 14 - 7):14, w_idx, p_idx]
        mean_demand = float(np.mean(recent_demands)) if len(recent_demands) > 0 else 20.0
        std_demand = float(np.std(recent_demands)) if len(recent_demands) > 1 else 5.0

        lead_time = 4.0  # nominal supplier lead time
        safety_factor = 1.65  # 95% service level
        safety_stock = round(safety_factor * std_demand * math.sqrt(lead_time), 1)
        reorder_point = round(mean_demand * lead_time + safety_stock, 1)

        # Inventory position = On-Hand + Pipeline
        inv_position = on_hand + pipeline

        if on_hand <= 0.0:
            stock_status = "CRITICAL_STOCKOUT"
            recommendation = f"Immediate expediting required: Deficit of {int(reorder_point - inv_position)} units."
        elif inv_position <= reorder_point:
            stock_status = "LOW_STOCK"
            order_qty = max(20, int(reorder_point - inv_position + safety_stock))
            recommendation = f"Reorder trigger reached: Issue replenishment order for {order_qty} units."
        elif on_hand > 2.5 * reorder_point:
            stock_status = "OVERSTOCKED"
            recommendation = "Stock surplus: Pause replenishment orders to reduce holding cost."
        else:
            stock_status = "NORMAL"
            recommendation = "Stock levels optimal within target buffer bounds."

        products_list.append(
            ProductInventoryItem(
                product_id=prod.id,
                product_name=prod.name,
                category=prod.category,
                unit_cost=float(prod.unit_cost),
                unit_price=float(prod.selling_price),
                on_hand_units=round(on_hand, 1),
                pipeline_units=round(pipeline, 1),
                recent_demand_rate=round(mean_demand, 1),
                reorder_point=reorder_point,
                safety_stock=safety_stock,
                fulfilled_units=round(fulfilled, 1),
                unmet_units=round(unmet, 1),
                stock_status=stock_status,
                recommendation=recommendation,
            )
        )

    tot_demand = total_fulfilled + total_unmet
    overall_fill_rate = round((total_fulfilled / tot_demand * 100.0), 2) if tot_demand > 0 else 100.0

    _cached_inventory_response = InventoryResponse(
        as_of_step=sim.current_step,
        total_on_hand_units=round(total_on_hand, 1),
        total_pipeline_units=round(total_pipeline, 1),
        total_fulfilled_demand=round(total_fulfilled, 1),
        total_unmet_demand=round(total_unmet, 1),
        overall_fill_rate_percent=overall_fill_rate,
        products=products_list,
        data_source=f"Digital Twin Simulator step {sim.current_step} (seed: 42, 14 operational days)",
    )

    return _cached_inventory_response
