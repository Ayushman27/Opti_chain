"""
Optimization recommendations route
POST /api/v1/optimization/recommendations
"""

from typing import List
from fastapi import APIRouter, HTTPException
import numpy as np

from optichain.api.schemas import (
    OptimizationRequest,
    OptimizationResponse,
    RecommendedOrder,
)
from optichain.api.dependencies import get_reference_simulation
from optichain.optimization.milp_planner import MILPReplenishmentPlanner

router = APIRouter(prefix="/optimization", tags=["Optimization"])


@router.post("/recommendations", response_model=OptimizationResponse)
def get_optimization_recommendations(req: OptimizationRequest) -> OptimizationResponse:
    """
    Formulates and solves the rolling-horizon constrained MILP replenishment problem
    using the PuLP/CBC mathematical programming solver.
    Returns optimal/feasible purchase order schedules, solver diagnostics, and solve time.
    """
    sim = get_reference_simulation(seed=42)
    sim.reset(seed=42)

    available_product_ids = {p.id: p for p in sim.products}
    available_supplier_map = {s.id: s for s in sim.suppliers}

    # Validate product_ids if provided
    if req.product_ids is not None:
        for pid in req.product_ids:
            if pid not in available_product_ids:
                raise HTTPException(
                    status_code=422,
                    detail=f"Product ID '{pid}' does not exist. Available products: {list(available_product_ids.keys())}"
                )

    H = req.horizon
    W = len(sim.warehouses)
    P = len(sim.products)

    # Build forecast demand matrix (H, W, P)
    # Using historical mean demand with safety buffer factor
    hist_matrix = getattr(sim, "historical_demand_matrix", None)
    if hist_matrix is not None:
        mean_demand = hist_matrix.mean(axis=0)  # Shape (W, P)
    else:
        mean_demand = sim.demand_matrix.mean(axis=0)

    forecast_demands = np.zeros((H, W, P), dtype=float)
    for tau in range(H):
        for w in range(W):
            for p in range(P):
                base_d = mean_demand[w, p]
                forecast_demands[tau, w, p] = max(0.0, float(base_d * (1.0 + req.safety_buffer_factor)))

    # Current on-hand inventory
    current_on_hand = sim.on_hand.copy()  # Shape (W, P)

    # In-transit arrivals across lookahead horizon
    scheduled_arrivals = np.zeros((H, W, P), dtype=float)

    # Initialize mathematical solver with user-specified timeout
    planner = MILPReplenishmentPlanner(time_limit_sec=float(req.time_limit_sec))

    try:
        opt_res = planner.solve(
            current_step=0,
            current_on_hand=current_on_hand,
            scheduled_arrivals=scheduled_arrivals,
            forecast_demands=forecast_demands,
            warehouses=sim.warehouses,
            products=sim.products,
            suppliers=sim.suppliers,
            horizon=H,
            transport_cost_per_unit=0.50,
            enable_moq=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MILP solver execution encountered an unexpected error: {str(e)}"
        )

    # Extract orders from full_schedule
    recommended_orders: List[RecommendedOrder] = []
    total_units = 0.0
    total_cost = 0.0

    target_pids = set(req.product_ids) if req.product_ids else set(available_product_ids.keys())

    for tau, schedule in opt_res.full_schedule.items():
        for (w_id, p_id, s_id), qty in schedule.items():
            if qty > 0 and p_id in target_pids:
                prod = available_product_ids[p_id]
                supplier = available_supplier_map.get(s_id)
                unit_cost = float(prod.unit_cost)
                line_cost = round(float(qty * unit_cost), 2)
                lead_time = int(supplier.base_lead_time_days) if supplier else 4

                recommended_orders.append(
                    RecommendedOrder(
                        step=tau,
                        product_id=p_id,
                        warehouse_id=w_id,
                        supplier_id=s_id,
                        quantity=float(qty),
                        unit_cost_usd=unit_cost,
                        estimated_cost_usd=line_cost,
                        lead_time_days=lead_time,
                    )
                )
                total_units += float(qty)
                total_cost += line_cost

    # Sort recommended orders chronologically by step then product
    recommended_orders.sort(key=lambda o: (o.step, o.product_id))

    return OptimizationResponse(
        solver_status=opt_res.status,
        termination_reason=opt_res.termination_reason,
        is_proven_optimal=opt_res.is_proven_optimal,
        is_feasible=opt_res.has_feasible_solution,
        solve_time_sec=round(opt_res.solve_time_sec, 4),
        planning_horizon_days=H,
        forecast_method_used=req.forecast_method,
        recommended_orders=recommended_orders,
        total_recommended_units=round(total_units, 1),
        total_estimated_cost_usd=round(total_cost, 2),
    )
